from app.agent import chain_with_history_and_agent

result = chain_with_history_and_agent.invoke({"input":"chao"},{'configurable': {'session_id': 'nibba'}})

print(result)