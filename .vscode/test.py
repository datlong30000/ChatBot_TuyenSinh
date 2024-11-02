#!/usr/bin/env python
"""
SERVER LANGCHAIN VỚI AGENT VÀ LỊCH SỬ CHAT (THỬ NGHIỆM)
"""
import re
from typing import Any, List, Union, Callable
from pathlib import Path
from fastapi import HTTPException  # Only if needed for tool execution
from langchain.memory import FileChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from fastapi import FastAPI
from langchain.agents import AgentExecutor, tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain_community.tools.convert_to_openai import format_tool_to_openai_tool
from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI

from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi.middleware.cors import CORSMiddleware
import logging
logger = logging.getLogger(__name__)
from dotenv import load_dotenv

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
        return FileChatMessageHistory(str(file_path))

    return get_chat_history

load_dotenv()
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

prompt = ChatPromptTemplate.from_messages(
    [
        (
        "system", 
         "you are college named Nguyễn Tất Thành answer only with Vietnamse language"
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
    ]
)


@tool
def get_answer(query: str) -> list:
    """use this tool if user ask something revelant about Nguyen Tat Thanh"""
    return retriever.get_relevant_documents(query)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, streaming=True)

tools = [get_answer]


llm_with_tools = llm.bind(tools=[format_tool_to_openai_tool(tool) for tool in tools])

#""" THỬ NGHIỆM """
#def prompt_trimmer(messages: List[Union[HumanMessage, AIMessage, FunctionMessage]]):
     #'''Trims the prompt to a reasonable length.'''
     # Keep in mind that when trimming you may want to keep the system message!
    # return messages[-10:] # Keep last 10 messages.

agent = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    #| prompt_trimmer # See comment above.
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

class Input(BaseModel):
    input: str
   
    chat_history: List[Union[HumanMessage, AIMessage, FunctionMessage]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "input", "output": "output"}},
    )


class Output(BaseModel):
    output: Any

chain_with_history_and_agent = RunnableWithMessageHistory(
    agent_executor,
    create_session_factory("chat_histories"),
    input_messages_key="input",
    history_messages_key="chat_history",
).with_types(input_type=Input)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using LangChain's Runnable interfaces",
)

# Allow requests from all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


add_routes(
    app,
    chain_with_history_and_agent,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)