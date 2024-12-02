from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
from pathlib import Path
from typing import Callable, Union, List, Optional
import re
import json
import logging
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class FileChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores history in a local file with a limit of 5 messages."""

    def __init__(
        self,
        file_path: str,
        *,
        encoding: Optional[str] = None,
        ensure_ascii: bool = True,
        message_limit: int = 5
    ) -> None:
        """Initialize the file path for the chat history.
        Args:
            file_path: The path to the local file to store the chat history.
            encoding: The encoding to use for file operations. Defaults to None.
            ensure_ascii: If True, escape non-ASCII in JSON. Defaults to True.
            message_limit: Maximum number of messages to keep. Defaults to 5.
        """
        self.file_path = Path(file_path)
        self.encoding = encoding
        self.ensure_ascii = ensure_ascii
        self.message_limit = message_limit

        if not self.file_path.exists():
            self.file_path.touch()
            self.file_path.write_text(
                json.dumps([], ensure_ascii=self.ensure_ascii), encoding=self.encoding
            )

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve the messages from the local file"""
        items = json.loads(self.file_path.read_text(encoding=self.encoding))
        # Keep only the most recent messages based on the limit
        items = items[-self.message_limit:] if items else []
        messages = messages_from_dict(items)
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in the local file, maintaining the message limit"""
        messages = messages_to_dict(self.messages)
        messages.append(messages_to_dict([message])[0])
        # Keep only the most recent messages based on the limit
        messages = messages[-self.message_limit:]
        self.file_path.write_text(
            json.dumps(messages, ensure_ascii=self.ensure_ascii), encoding=self.encoding
        )

    def clear(self) -> None:
        """Clear session memory from the local file"""
        self.file_path.write_text(
            json.dumps([], ensure_ascii=self.ensure_ascii), encoding=self.encoding
        )

def _is_valid_identifier(value: str) -> bool:
    """Check if the session ID is in a valid format."""
    # Use a regular expression to match the allowed characters
    valid_characters = re.compile(r"^[a-zA-Z0-9-_]+$")
    return bool(valid_characters.match(value))


def create_session_factory(
    base_dir: Union[str, Path],
) -> Callable[[str], BaseChatMessageHistory]:
    """Create a session ID factory that creates session IDs from a base dir.

    Args:
        base_dir: Base directory to use for storing the chat histories.

    Returns:
        A session ID factory that creates session IDs from a base path.
    """
    base_dir_ = Path(base_dir) if isinstance(base_dir, str) else base_dir
    if not base_dir_.exists():
        base_dir_.mkdir(parents=True)

    def get_chat_history(session_id: str) -> FileChatMessageHistory:
        """Get a chat history from a session ID."""
        if not _is_valid_identifier(session_id):
            raise HTTPException(
                status_code=400,
                detail=f"Session ID `{session_id}` is not in a valid format. "
                        "Session ID must only contain alphanumeric characters, "
                        "hyphens, and underscores.",
            )
        logger.info(f"Received request with session ID: {session_id}")
        file_path = base_dir_ / f"{session_id}.json"
        return FileChatMessageHistory(str(file_path), encoding="utf-8", message_limit=8)

    return get_chat_history