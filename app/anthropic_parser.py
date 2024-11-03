from typing import List, Union, Dict, Any
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatGeneration, Generation
from langchain.agents.agent import MultiActionAgentOutputParser
from langchain.agents.output_parsers.tools import (
    ToolAgentAction,
    parse_ai_message_to_tool_action,
)

AnthropicToolAgentAction = ToolAgentAction

def parse_ai_message_to_Anthropic_tool_action(
    message: BaseMessage,
) -> Union[List[AgentAction], AgentFinish]:
    """Parse an AI message containing text content into AgentFinish."""
    # If the message is in the expected format with text content
    if isinstance(message, AIMessage) and hasattr(message, 'content'):
        content = message.content
        # If content is a list of dictionaries with text field
        if isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict):
            text = content[0].get('text', '')
            return AgentFinish(
                return_values={"output": text},
                log=text
            )
    
    # Fallback to original tool parsing logic if format doesn't match
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
    Handles both tool calls and direct text responses in the format:
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
        return parse_ai_message_to_Anthropic_tool_action(message)

    def parse(self, text: str) -> Union[List[AgentAction], AgentFinish]:
        """Parse a text string into an AgentFinish."""
        try:
            # Assuming the text is a string representation of a list with a dictionary
            # Convert it to the expected format
            return AgentFinish(
                return_values={"output": text},
                log=text
            )
        except:
            raise ValueError("Invalid format. Expected text content.")