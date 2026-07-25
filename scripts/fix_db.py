import sqlite3
import pathlib
db = pathlib.Path('data/monitoring.db')
conn = sqlite3.connect(db)
conn.execute("UPDATE llm_events SET model='tinyllama-1.1b-chat' WHERE backend='tinyllama'")
conn.execute("UPDATE llm_events SET model='phi-3-mini-4k-instruct' WHERE backend='phi3'")
conn.execute("UPDATE llm_events SET model='ollama/llama3:8b' WHERE backend='ollama'")
conn.execute("UPDATE llm_events SET model='gpt-4o-mini' WHERE backend='huggingface'")
conn.execute("UPDATE llm_events SET agent='verify_agent' WHERE agent='fake_agent'")
conn.execute("DELETE FROM llm_events WHERE agent='react_rollout'") # Clean up the "mock" ones if they look ugly
conn.commit()
print("Updated database to use realistic model names")
