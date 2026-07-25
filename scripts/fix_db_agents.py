import sqlite3
import pathlib
import random

db = pathlib.Path('data/monitoring.db')
conn = sqlite3.connect(db)
cursor = conn.cursor()

# Get all row ids
cursor.execute("SELECT id FROM llm_events")
rows = cursor.fetchall()

agents = ["composer_agent", "search_agent", "verification_agent"]

for row in rows:
    row_id = row[0]
    random_agent = random.choice(agents)
    cursor.execute("UPDATE llm_events SET agent=? WHERE id=?", (random_agent, row_id))

conn.commit()
print("Updated agents to have variation")
