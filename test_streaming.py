import httpx
import asyncio
import json
from typing import Dict, AsyncGenerator

async def parse_stream_events(
    endpoint_url: str,
    input_text: str,
    session_id: str
) -> AsyncGenerator[Dict, None]:
    """
    Streams and parses events from the API endpoint, yielding parsed events in the same
    format as the original chain_with_history_and_agent implementation.
    """
    payload = {
        "input": {
            "input": input_text
        },
        "config": {
            "configurable": {
                "session_id": session_id
            }
        }
    }

    async with httpx.AsyncClient() as client:
        try:
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream'
            }
            
            async with client.stream('POST', endpoint_url, json=payload, headers=headers) as response:
                buffer = ""
                async for raw_bytes in response.aiter_bytes():
                    try:
                        text = raw_bytes.decode('utf-8')
                        buffer += text
                        
                        # Process complete lines in buffer
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            
                            if line.startswith('data: '):
                                try:
                                    event_data = json.loads(line[6:])
                                    yield event_data
                                except json.JSONDecodeError as e:
                                    print(f"Error parsing JSON: {str(e)}")
                                    print(f"Invalid JSON line: {line[6:]}")

                    except Exception as e:
                        print(f"Error processing response: {str(e)}")
                        print(f"Raw bytes: {raw_bytes}")

        except httpx.HTTPError as e:
            print(f"HTTP error occurred: {str(e)}")
            if hasattr(e, 'response'):
                print(f"Error response text: {await e.response.aread()}")

async def process_events(event: Dict):
    """
    Processes and logs events in the same format as the original implementation.
    """
    kind = event.get("event")
    
    if kind == "on_chain_start":
        if event["name"] == "agent":
            print(
                f"Starting agent: {event['name']} with input: {event['data'].get('input')}"
            )

    elif kind == "on_chain_end":
        if event["name"] == "agent":
            print("\n--")
            print(
                f"Done agent: {event['name']} with output: {event['data'].get('output', {}).get('output')}"
            )

    elif kind == "on_chat_model_stream":
        content = event.get("data", {}).get("chunk", {}).get("content")
        if content:
            print(content, end="")

    elif kind == "on_tool_start":
        print("--")
        print(
            f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
        )

    elif kind == "on_tool_end":
        print(f"Done tool: {event['name']}")
        print(f"Tool output was: {event['data'].get('output')}")
        print("--")

async def run_stream():
    endpoint = "http://127.0.0.1:8000/stream_events"  # Update with your endpoint
    input_text = "viet doan van 700 tu ve truong nttu di"
    session_id = "900"
    
    print(f"Connecting to endpoint: {endpoint}")
    
    async for event in parse_stream_events(
        endpoint_url=endpoint,
        input_text=input_text,
        session_id=session_id
    ):
        await process_events(event)

if __name__ == "__main__":
    asyncio.run(run_stream())