from app.agent import chain_with_history_and_agent

result = chain_with_history_and_agent.invoke({"input":"hay viet doan van 500 tu ve truong nttu di"},{'configurable': {'session_id': 'nibba'}})