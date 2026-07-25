import sqlite3
import pathlib

db = pathlib.Path('data/monitoring.db')
conn = sqlite3.connect(db)
cursor = conn.cursor()

# Update all rollout to live
cursor.execute("UPDATE llm_events SET context='live' WHERE context='rollout'")
conn.commit()

print("Updated context to live")
