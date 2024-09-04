
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents import AgentExecutor, tool
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from langserve.pydantic_v1 import BaseModel, Field
from typing import Any
from langchain_core.runnables import ConfigurableFieldSpec
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()



embeddings = OpenAIEmbeddings()
retriever_de_an = FAISS.load_local("vector_database/faiss_index_de_an", embeddings, allow_dangerous_deserialization=True).as_retriever()
