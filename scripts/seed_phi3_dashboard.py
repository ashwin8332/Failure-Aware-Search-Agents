"""
Run Phi-3-mini queries through the full agent pipeline AND log them
directly to the monitoring store so the dashboard shows phi3 data.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.store import get_store
from shared.phi3_llm import Phi3LLM
from shared.search_tools import web_search, search_quality_check, wikipedia_lookup
from shared.rewards import extract_confidence

QUERIES = [
    "What is the capital of France?",
    "Who invented the telephone?",
    "What drug treats both malaria and lupus?",
    "Who became president after JFK?",
    "What causes the northern lights?",
    "Is coffee good or bad for health?",
    "What is the speed of light?",
    "Who wrote Romeo and Juliet?",
]

print("Loading Phi-3-mini-4k-instruct (from cache)...")
llm = Phi3LLM(max_new_tokens=200)
print("Model loaded!\n")
print("-" * 65)

store = get_store()

for query in QUERIES:
    t0 = time.perf_counter()

    # 1. Real web search
    result = web_search(query, 5)
    snippets = [r.get("snippet", "") for r in result.get("results", [])]

    # 2. Search quality check
    quality = search_quality_check(query, snippets)
    policy_hint = quality["recommendation"]

    # 3. Phi-3 generates the answer
    phi3_prompt = (
        f"Question: {query}\n"
        f"Retrieved snippets: {snippets[:2]}\n"
        f"Search quality: {quality['quality_score']:.2f}\n"
        "Give a direct, concise answer with a confidence percentage."
    )
    answer = llm.generate(phi3_prompt, temperature=0.6)

    # 4. Extract confidence from answer
    confidence = extract_confidence(answer)
    if policy_hint == "confident":
        confidence = max(confidence, 0.65)
    elif policy_hint == "re_search":
        confidence = min(confidence, 0.40)

    latency_ms = (time.perf_counter() - t0) * 1000

    # 5. Record as a full query event (shows in dashboard queries table)
    store.record_query(
        query=query,
        policy=policy_hint,
        confidence=confidence,
        latency_ms=latency_ms,
        retrieval_quality=quality["quality_score"],
        status="ok",
    )

    policy_label = {"confident": "CONFIDENT", "hedge": "HEDGE", "re_search": "RE_SEARCH"}.get(policy_hint, policy_hint.upper())
    print(f"[{policy_label:10}] {int(confidence*100):3}%  {query}")
    print(f"  Phi-3: {answer[:120].strip()}...")
    print()

print("-" * 65)
print(f"Done! {len(QUERIES)} Phi-3 queries logged.")
print("Refresh http://localhost:8080/dashboard")
