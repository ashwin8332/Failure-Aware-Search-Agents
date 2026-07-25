import httpx, json, time

queries = [
    "What is the capital of France?",
    "Who invented the telephone?",
    "What drug treats both malaria and lupus?",
    "Who became president after JFK?",
    "What causes the northern lights?",
    "Is coffee good or bad for health?",
    "What is the speed of light?",
    "Who wrote Romeo and Juliet?",
]

print("Seeding dashboard — sending queries through the full agent pipeline...")
print("-" * 65)
for q in queries:
    try:
        r = httpx.post("http://localhost:8080/query", json={"query": q}, timeout=60)
        d = r.json()
        policy = d.get("policy", "?").upper()
        conf   = int(d.get("confidence", 0) * 100)
        print(f"  [{policy:10}] {conf:3}%  {q[:48]}")
    except Exception as e:
        print(f"  [ERROR]  {e}  {q[:48]}")
    time.sleep(0.5)

print("-" * 65)
print("Done! Refresh http://localhost:8080/dashboard")
