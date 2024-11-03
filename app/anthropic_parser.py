from typing import List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, Generation
from langchain.agents.agent import MultiActionAgentOutputParser
from langchain.agents.output_parsers.tools import (
    ToolAgentAction,
    parse_ai_message_to_tool_action,
)

AnthropicToolAgentAction = ToolAgentAction

def parse_ai_message_to_anthropic_tool_action(
    message: BaseMessage,
) -> Union[List[AgentAction], AgentFinish]:
    """Parse an AI message potentially containing tool_calls or text output."""
    # Check for text output format first
    content = message.content
    if isinstance(content, list) and len(content) > 0:
        if isinstance(content[0], dict) and 'text' in content[0]:
            return AgentFinish(
                return_values={"output": content[0]['text']},
                log=content[0]['text']
            )
    
    # Fall back to original tool parsing if not in text format
    tool_actions = parse_ai_message_to_tool_action(message)
    if isinstance(tool_actions, AgentFinish):
        return tool_actions
    
    final_actions: List[AgentAction] = []
    for action in tool_actions:
        if isinstance(action, ToolAgentAction):
            final_actions.append(
                AnthropicToolAgentAction(
                    tool=action.tool,
                    tool_input=action.tool_input,
                    log=action.log,
                    message_log=action.message_log,
                    tool_call_id=action.tool_call_id,
                )
            )
        else:
            final_actions.append(action)
    return final_actions

class AnthropicToolsAgentOutputParser(MultiActionAgentOutputParser):
    """Parses a message into agent actions/finish.
    Handles both tool calls and text output in the format:
    [{'text': "...", 'type': 'text', 'index': 0}]
    """

    @property
    def _type(self) -> str:
        return "Anthropic-tools-agent-output-parser"

    def parse_result(
        self, result: List[Generation], *, partial: bool = False
    ) -> Union[List[AgentAction], AgentFinish]:
        if not isinstance(result[0], ChatGeneration):
            raise ValueError("This output parser only works on ChatGeneration output")
        message = result[0].message
        return parse_ai_message_to_anthropic_tool_action(message)

    def parse(self, text: str) -> Union[List[AgentAction], AgentFinish]:
        raise ValueError("Can only parse messages")