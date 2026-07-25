# Failure-Aware Search Agents

Post-train LLM search agents with **Direct Preference Optimization (DPO)** so they express calibrated uncertainty and trigger re-search under ambiguous retrieval — instead of hallucinating confidently.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     OFFLINE TRAINING PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────┤
│  HotpotQA / TriviaQA / AmbigQA  →  ReAct rollouts (k=8)  →  scoring    │
│       ↓                              ↓                        ↓       │
│  ambiguous_queries.parquet     trajectories JSONL      composite reward │
│       ↓                              ↓                        ↓       │
│                         DPO preference pairs  →  QLoRA DPOTrainer       │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                     LIVE INFERENCE (ADK + MCP)                          │
├─────────────────────────────────────────────────────────────────────────┤
│  User Query → Orchestrator (SequentialAgent)                            │
│                 ├─ Search Agent (web_search, quality_check)             │
│                 ├─ Verification Agent (wikipedia, coherence)            │
│                 └─ Response Composer (confident | hedge | re_search)    │
│  MCP Server: FastMCP stdio tools with rate limiting + sanitization      │
└─────────────────────────────────────────────────────────────────────────┘
```

### Three inference outcomes

| Policy | When | Behavior |
|--------|------|----------|
| **confident** | Verification confirmed, good retrieval | Clear answer, 70–90% confidence |
| **hedge** | Partial / unverified evidence | Explicit uncertainty, 40–65% |
| **re_search** | Conflicts or insufficient evidence | Explain conflict, recommend re-query, <40% |

### Composite reward (training)

```
R(τ) = 0.5·R_fact + 0.3·R_calib + 0.2·R_recovery
```

- **R_fact**: BERTScore F1 vs gold (token F1 fallback)
- **R_calib**: `1 - |confidence - correctness|`
- **R_recovery**: bonus for re-search when retrieval quality is low

---

## Repo structure

```
├── app/                    # FastAPI inference API
├── agents/                 # ADK orchestrator + sub-agents + composer
├── training/               # DPO fine-tuning (TRL + PEFT + QLoRA)
├── mcp_server/             # FastMCP search tools
├── scripts/                # Dataset, rollouts, scoring, preference pairs
├── evaluation/             # Baseline vs aligned metrics
├── shared/                 # Search tools, rewards, schemas
├── configs/                # YAML configs for training & eval
├── prompts/                # Agent system prompts
├── examples/               # Sample prompts & outputs
├── data/                   # Generated datasets (gitignored artifacts)
└── results/                # Evaluation reports
```

---

## Quick start (local)

### 1. Environment

**All run commands:** see **[run.md](run.md)** — includes [Quick start](run.md#quick-start-first-time), dashboard, Ollama, CPU/GPU workflows.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

**Environment variables** (see `.env.example`):

| Variable | Required | Purpose |
|----------|----------|---------|
| `GOOGLE_API_KEY` | For ADK/Gemini live agents | Google ADK inference |
| `SERPAPI_KEY` | Optional | Web search (falls back to DuckDuckGo/mock) |
| `HF_TOKEN` | Optional | Gated HuggingFace models |
| `WANDB_API_KEY` | Optional | Training logs |
| `BASE_MODEL` | Optional | Default `mistralai/Mistral-7B-Instruct-v0.3` |

### 2. Offline pipeline (CPU-friendly with mock LLM)

```bash
# Build ambiguous query dataset
python scripts/generate_dataset.py --target-size 200

# Generate k=8 rollouts per query (mock LLM, no GPU)
python scripts/generate_rollouts.py --max-queries 20 --k 8

# Score trajectories
python scripts/score_trajectories.py

# Build DPO preference pairs
python scripts/build_preference_pairs.py
```

### 3. DPO training (GPU recommended)

```bash
python training/dpo_train.py --epochs 1 --max_samples 10   # smoke test
python training/dpo_train.py --epochs 3                   # full run
```

Checkpoints saved to `checkpoints/dpo-failure-aware/`.

### 4. Live inference

**Mock orchestrator** (no API keys):

```bash
python agents/orchestrator.py --query "What is the capital of France?"
```

**FastAPI**:

```bash
python -m app.main
# Dashboard: http://localhost:8080/dashboard
# POST http://localhost:8080/query  {"query": "..."}
```

**Google ADK** (requires `GOOGLE_API_KEY`):

```bash
python agents/orchestrator.py --query "..." --adk
# or: adk web --port 8080  (with agents module on PYTHONPATH)
```

### 4.1 What server / GPU are we using?

The repository itself does not hard-code a physical server. It runs wherever Python is started.

The FastAPI `/health` endpoint now reports runtime environment details, including:
- host name
- platform details
- Python version
- whether this is detected as a Kaggle environment
- GPU availability and device names

Call:

```bash
curl http://localhost:8080/health
```

If you are on a local machine, the service uses the same CPU/GPU available to your Python runtime.

If you are on Kaggle, the service can detect the Kaggle environment and tell you so.

### 4.2 CPU vs GPU behavior

The code supports three kinds of model backends:
- `mock` — no real LLM, no GPU needed
- `ollama` — remote Ollama server, can use local CPU or GPU if Ollama is configured that way
- `adk` — Google ADK backend, independent of this repo's GPU

For HuggingFace backends (`shared/phi3_llm.py`, `shared/tinyllama_llm.py`), the library auto-detects GPU via `torch.cuda.is_available()` and uses GPU if available.

### 4.3 How to deploy and run on Kaggle

On Kaggle, use the notebook / script environment and then run this repo in the kernel.

1. Upload the repo to Kaggle or sync it via Git.
2. Create a Kaggle notebook with a Python 3 environment.
3. Enable the GPU accelerator in Kaggle notebook settings if available.
4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Set environment variables in the Kaggle notebook, if needed:

```python
import os
os.environ["INFERENCE_BACKEND"] = "ollama"  # or "mock"
os.environ["OLLAMA_MODEL"] = "phi3:mini"
```

6. Run the service locally in the Kaggle kernel:

```bash
python -m app.main
```

7. Check runtime detection:

```bash
curl http://127.0.0.1:8080/health
```

8. Query the service:

```bash
curl -X POST http://127.0.0.1:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Who invented the telephone?", "mode": "general"}'
```

> Note: Kaggle may not expose an external port for public access. Use the notebook kernel or Kaggle UI tools to send requests from inside the same session.

#### Kaggle helper script

You can use the included helper script to automate the common steps in a Kaggle kernel or VM. It installs dependencies, starts the FastAPI server, checks `/health`, runs a sample `/query`, and tails the server logs.

Run the script from the repository root in a terminal cell:

```bash
bash scripts/kaggle_run.sh
```

If you prefer to run interactively from the notebook, use the notebook at `notebooks/kaggle_demo.ipynb` which performs the same steps and shows how to inspect runtime detection (GPU/Kaggle).

### 4.4 If no GPU is available

If Kaggle or local runtime has no GPU, use `mock` or `ollama` with a CPU-only Ollama server.

- For inference with no GPU: `INFERENCE_BACKEND=mock`
- For local HuggingFace backends that can use CPU: the code will fall back to CPU automatically

### 4.5 Recommended deployment path for real data

1. Start with `mock` to validate routes and metrics.
2. Use `ollama` if you have a local Ollama model server on the same machine.
3. For production-quality LLMs and GPU resources, use `adk` only if you have Google ADK credentials configured.

---

### 5. Evaluation

```bash
python evaluation/run_eval.py --max-queries 20 --condition all
```

Reports: ECE, hallucination rate, task F1, re-search rate on low-quality retrieval.

---

## Docker

```bash
docker compose up api          # FastAPI on :8080
docker compose up mcp-server   # MCP stdio server
docker compose --profile train up training  # smoke DPO train
```

---

## MCP tools

| Tool | Description |
|------|-------------|
| `web_search(query, num_results)` | SerpAPI → DuckDuckGo → mock fallback |
| `wikipedia_lookup(entity)` | Wikipedia summary |
| `coherence_score(texts)` | Conflict detection across snippets |
| `search_quality_check(query, snippets)` | Returns `confident` / `hedge` / `re_search` |

Security: input sanitization (500 char max), token-bucket rate limit (10/min default).

---

## Design notes

- **Not prompt-only**: preference pairs encode calibrated uncertainty + recovery behavior; DPO shifts the policy toward chosen trajectories.
- **Mocks everywhere**: dataset, rollouts, search, and orchestrator work without paid APIs or GPU.
- **Production path**: swap `MockLLM` → fine-tuned checkpoint, `MockOrchestrator` → ADK `SequentialAgent` + MCP.

---

## References

- Research spec: `failure_aware_search_agents_research.pdf`
- Implementation spec: `failure_aware_agent_implementation_prompt.md`
- Base model: [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

---

*Kaggle AI Agents Hackathon 2025 — Failure-Aware Search Agents*
