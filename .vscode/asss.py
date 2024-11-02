#!/usr/bin/env python
"""Example of a chat server with persistence handled on the backend.

For simplicity, we're using file storage here -- to avoid the need to set up
a database. This is obviously not a good idea for a production environment,
but will help us to demonstrate the RunnableWithMessageHistory interface.

We'll use cookies to identify the user and/or session. This will help illustrate how to
fetch configuration from the request.
"""
from langchain.agents import tool
import re
from pathlib import Path
from typing import Callable, Union

from fastapi import FastAPI, HTTPException
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import FileChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.tools.convert_to_openai import format_tool_to_openai_tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv

load_dotenv()

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
        file_path = base_dir_ / f"{session_id}.json"
        return FileChatMessageHistory(str(file_path))

    return get_chat_history

xlsx_loader = DirectoryLoader('C:/Users/Huynhlong/AI_lord/data', glob="**/*.xlsx")
xlsx_docs = xlsx_loader.load()

# DirectoryLoader cho file .txt
text_loader = DirectoryLoader('C:/Users/Huynhlong/AI_lord/data', glob="**/*.txt")
text_docs = text_loader.load()

# Kết hợp dữ liệu từ cả hai loại loader
docs = text_docs + xlsx_docs

embeddings = OpenAIEmbeddings()


text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

@tool
def get_answer(query: str) -> list:
    """use this tool if user ask something revelant about Nguyen Tat Thanh"""
    return retriever.get_relevant_documents(query)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, streaming=True)

tools = [get_answer]


llm_with_tools = llm.bind(tools=[format_tool_to_openai_tool(tool) for tool in tools])

prompt = ChatPromptTemplate.from_messages(
    [
        (
        "system", 
         "you are college named Nguyễn Tất Thành answer only with Vietnamse language"
         "you are a wise and helpful chat bot that can talk like a normal person with empathy and logical"
         
        ),
        # Please note the ordering of the fields in the prompt!
        # The correct ordering is:
        # 1. history - the past messages between the user and the agent
        # 2. user - the user's current input
        # 3. agent_scratchpad - the agent's working space for thinking and
        #    invoking tools to respond to the user's input.
        # If you change the ordering, the agent will not work correctly since
        # the messages will be shown to the underlying LLM in the wrong order.
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    #| prompt_trimmer # See comment above.
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

# Declare a chain

chain = prompt | agent | ChatOpenAI(model="gpt-3.5-turbo")


class InputChat(BaseModel):
    """Input for the chat endpoint."""

    # The field extra defines a chat widget.
    # As of 2024-02-05, this chat widget is not fully supported.
    # It's included in documentation to show how it should be specified, but
    # will not work until the widget is fully supported for history persistence
    # on the backend.
    human_input: str = Field(
        ...,
        description="The human input to the chat system.",
        extra={"widget": {"type": "chat", "input": "human_input"}},
    )


chain_with_history = RunnableWithMessageHistory(
  chain,
  create_session_factory("chat_histories"),
  input_messages_key="human_input",
  history_messages_key="history",
  tools=tools
).with_types(input_type=InputChat)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    chain_with_history,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)