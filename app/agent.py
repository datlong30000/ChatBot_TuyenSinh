from session_verify import create_session_factory
from prompt_templates import prompt
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.agents.output_parsers import ToolsAgentOutputParser
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from anthropic_parser import AnthropicToolsAgentOutputParser
from langchain_anthropic.chat_models import convert_to_anthropic_tool
from pydantic import BaseModel, Field
from typing import Any
from dotenv import load_dotenv
from tools import *

# Áp dụng các biến môi trường
load_dotenv()

# # Lựa chọn mô hình LLM
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, streaming=True)

llm = ChatAnthropic(
    model_name="claude-3-5-sonnet-latest",
    temperature=0.5,
    max_tokens_to_sample=2000,
    timeout=None,
    max_retries=7,
    streaming=False,
    verbose=False,
    # other params...
)

# Khởi tạo tool
tools = [tavily_search_web, de_an_tools, diem_trung_tuyen_tools, hoc_phi_nganh_tools, nang_khieu_tools,
         pt_hoc_ba_tools, pt_nang_luc_tools, pt_uu_tien_tools, nganh_hoc_tools, 
           tinh_hinh_viec_lam_tools, thoi_gian_het_han_tools,thoi_gian_hien_tai, xin_nghi_tools]


# # Kết hợp LLM với tools
# llm_with_tools = llm.bind(tools=[convert_to_openai_tool(tool) for tool in tools])
llm_with_tools = llm.bind(tools=[convert_to_anthropic_tool(tool) for tool in tools])

# Tạo Agent Executor
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_tools
    | AnthropicToolsAgentOutputParser()
)

# agent = create_tool_calling_agent(llm_with_tools, tools, prompt=prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

class Input(BaseModel):
    input: str = Field(
    
        ...,
        description="The human input to the chat system.",
        extra={"widget": {"type": "chat", "input": "input", "output":"output"}},
    )



class Output(BaseModel):
    output: Any

chain_with_history_and_agent = RunnableWithMessageHistory(
    agent_executor,
    create_session_factory("chat_histories"),
    input_messages_key="input",
    history_messages_key="chat_history",
).with_types(input_type=Input, output_type=Output)
