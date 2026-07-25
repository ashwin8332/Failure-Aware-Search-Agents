# Run Commands — Failure-Aware Search Agents

All commands for **Windows** and **Linux/macOS**: CPU vs GPU, real vs sample data, mock / Ollama / HuggingFace LLMs, inference API, and the **LLM monitoring dashboard**.

> **Tip:** On Windows, if `python` is not on PATH, use `py` instead (e.g. `py -m venv .venv`).

---

## Quick start (first time)

Follow these steps to go from zero to a running project in ~10 minutes (CPU, no API keys).

### Step 1 — Clone & enter the repo

```bash
cd Failure-Aware-Search-Agents
```

### Step 2 — Create venv & install deps

**Windows (PowerShell)**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
copy .env.example .env
$env:PYTHONPATH = (Get-Location).Path
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
```

### Step 3 — Run the fastest smoke test (CPU, sample data, mock LLM)

```bash
python scripts/run_offline_pipeline.py --target-size 200 --max-queries 10 --k 4 --synthetic-only
```

Creates under `data/`:
- `ambiguous_queries.parquet` — synthetic ambiguous questions
- `rollouts/` — mock agent trajectories
- `scored_trajectories.jsonl` — reward scores
- `preference_pairs/` — DPO training pairs

> Scoring downloads BERTScore (`roberta-large`) and can take several minutes on CPU.

### Step 4 — Start the API + monitoring dashboard

```bash
python -m app.main
```

Open in your browser:

| URL | What it does |
|-----|----------------|
| **http://localhost:8080/dashboard** | LLM monitoring UI (live + offline metrics) |
| http://localhost:8080/health | Health check |
| http://localhost:8080/query | Inference API (POST JSON) |

### Step 5 — Send a test query

**Windows (PowerShell)**

```powershell
Invoke-RestMethod -Method POST -Uri "http://localhost:8080/query" `
  -ContentType "application/json" `
  -Body '{"query":"What is the capital of France?"}'
```

**Linux / macOS**

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'
```

Refresh the dashboard — you should see latency, policy (`confident` / `hedge` / `re_search`), and confidence.

### Step 6 (optional) — Ollama rollouts

If [Ollama](https://ollama.com) is installed with `phi3:mini` or `tinyllama`:

```bash
python scripts/generate_rollouts.py --max-queries 3 --k 2 --ollama-model phi3:mini
```

Ollama calls appear in the dashboard **LLM calls** table and in `data/monitoring.db`.

---

## What's in this project (recent additions)

| Feature | Location | Notes |
|---------|----------|-------|
| **LLM monitoring dashboard** | `http://localhost:8080/dashboard` | Live query metrics + offline reward charts |
| **Monitoring store** | `data/monitoring.db` | SQLite — query events & LLM call logs |
| **Ollama rollouts** | `--ollama-model phi3:mini` | Local LLM via `shared/ollama_llm.py` |
| **Synthetic sample data** | `--synthetic-only` | Skip HF dataset download |
| **Mock orchestrator fix** | `agents/orchestrator.py` | Works with google-adk installed (uses mocks when no API key) |
| **Run wrappers** | `run.ps1`, `run.sh` | Optional shortcuts — this file is the full reference |

**Dashboard tracks:**
- Live `POST /query` — latency, policy, confidence, retrieval quality
- LLM backends — `ollama`, `huggingface`, `mock` (from rollouts)
- Offline — composite reward, calibration histograms from `data/scored_trajectories.jsonl`
- Eval — ECE, hallucination rate from `results/eval_results.json` (after `run_eval.py`)

---

## Quick matrix

| Goal | Data | LLM | Hardware | Section |
|------|------|-----|----------|---------|
| **First run (recommended)** | Sample/synthetic | Mock | CPU | [Quick start](#quick-start-first-time) |
| Fastest smoke test | Sample/synthetic | Mock | CPU | [CPU + sample + mock LLM](#3-cpu--sample-data--mock-llm) |
| Real QA datasets | HotpotQA / TriviaQA / AmbigQA | Mock | CPU + network | [CPU + real data](#4-cpu--real-huggingface-data) |
| Local LLM rollouts | Either | Ollama | CPU/GPU via Ollama | [Ollama](#5-ollama-local-llm) |
| HF model rollouts | Either | HuggingFace | GPU recommended | [HuggingFace LLM](#6-huggingface-llm-rollouts) |
| DPO fine-tuning | Preference pairs | Mistral-7B QLoRA | GPU required | [GPU DPO training](#7-gpu-dpo-training) |
| Live query + monitor | — | Mock / Gemini ADK | CPU | [Inference & dashboard](#8-inference) |

---

## 1. One-time setup

### Windows (PowerShell)

```powershell
cd C:\path\to\Failure-Aware-Search-Agents
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
copy .env.example .env
$env:PYTHONPATH = (Get-Location).Path
```

### Linux / macOS

```bash
cd /path/to/Failure-Aware-Search-Agents
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
```

Activate venv on every new terminal:

| Windows | Linux / macOS |
|---------|---------------|
| `.\.venv\Scripts\Activate.ps1` | `source .venv/bin/activate` |

---

## 2. Ollama models

Install [Ollama](https://ollama.com), then pull models. You currently have **tinyllama** and **phi3:mini**; add more as needed.

```bash
# Same on Windows, Linux, macOS
ollama list
ollama pull tinyllama
ollama pull phi3:mini

# Optional — uncomment when you want them
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
ollama pull mistral:7b
```

Set in `.env` (optional):

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=phi3:mini
```

---

## 3. CPU + sample data + mock LLM

**Best for:** no GPU, no Ollama, no HuggingFace dataset download. Uses built-in synthetic queries and a deterministic mock LLM.

### Full offline pipeline (dataset → rollouts → score → preference pairs)

**Windows**

```powershell
python scripts/run_offline_pipeline.py --target-size 200 --max-queries 10 --k 4 --synthetic-only
```

**Linux / macOS**

```bash
python scripts/run_offline_pipeline.py --target-size 200 --max-queries 10 --k 4 --synthetic-only
```

### Step by step

| Step | Windows & Linux (same command) |
|------|--------------------------------|
| 1. Sample dataset | `python scripts/generate_dataset.py --target-size 200 --synthetic-only` |
| 2. Mock rollouts | `python scripts/generate_rollouts.py --max-queries 10 --k 4` |
| 3. Score trajectories | `python scripts/score_trajectories.py` |
| 4. DPO preference pairs | `python scripts/build_preference_pairs.py` |

> **Note:** Scoring uses BERTScore (`roberta-large`) and is slow on CPU. Keep `--max-queries` and `--k` small for quick tests.

---

## 4. CPU + real HuggingFace data

**Best for:** real ambiguous QA data from HotpotQA, TriviaQA, AmbigQA. Needs internet; first run downloads datasets.

### Real dataset only

**Windows & Linux**

```bash
python scripts/generate_dataset.py --target-size 500
```

Outputs: `data/ambiguous_queries.parquet`, `data/ambiguous_test_set.parquet`

### Full pipeline with real data + mock LLM

**Windows & Linux**

```bash
python scripts/run_offline_pipeline.py --target-size 500 --max-queries 20 --k 8
```

### Real data, step by step (mock LLM rollouts)

```bash
python scripts/generate_dataset.py --target-size 500
python scripts/generate_rollouts.py --max-queries 20 --k 8
python scripts/score_trajectories.py
python scripts/build_preference_pairs.py
```

---

## 5. Ollama (local LLM)

**Best for:** real LLM rollouts on your machine. Ollama handles CPU/GPU; no HuggingFace model download for rollouts.

### Rollouts only (needs existing `data/ambiguous_queries.parquet`)

**phi3:mini**

```bash
python scripts/generate_rollouts.py --max-queries 5 --k 2 --ollama-model phi3:mini
```

**tinyllama**

```bash
python scripts/generate_rollouts.py --max-queries 5 --k 2 --ollama-model tinyllama
```

**qwen2.5:3b** (after `ollama pull qwen2.5:3b`)

```bash
python scripts/generate_rollouts.py --max-queries 5 --k 2 --ollama-model qwen2.5:3b
```

### Full pipeline: sample data + Ollama rollouts

```bash
python scripts/run_offline_pipeline.py \
  --target-size 50 --max-queries 5 --k 2 \
  --synthetic-only --ollama-model phi3:mini
```

### Full pipeline: real data + Ollama rollouts

```bash
python scripts/generate_dataset.py --target-size 500
python scripts/run_offline_pipeline.py \
  --target-size 500 --max-queries 20 --k 8 \
  --ollama-model phi3:mini
```

> `run_offline_pipeline.py` passes `--ollama-model` only to the rollout step. Generate real data first (no `--synthetic-only`), or use the two-command sequence above.

---

## 6. HuggingFace LLM rollouts

**Best for:** rollouts with a HF model ID. **GPU strongly recommended**; CPU works but is very slow.

```bash
# Example — small instruct model on GPU
python scripts/generate_rollouts.py \
  --max-queries 5 --k 2 \
  --model-id microsoft/Phi-3-mini-4k-instruct

# Default training base model (7B — needs GPU + VRAM)
python scripts/generate_rollouts.py \
  --max-queries 5 --k 2 \
  --model-id mistralai/Mistral-7B-Instruct-v0.3
```

Omit both `--model-id` and `--ollama-model` to use the **mock LLM** (CPU-friendly).

---

## 7. GPU DPO training

**Best for:** fine-tuning after preference pairs exist in `data/preference_pairs/`. Requires CUDA GPU.

### Smoke test (1 epoch, 10 samples)

**Windows & Linux**

```bash
python training/dpo_train.py --epochs 1 --max_samples 10
```

### Full run

```bash
python training/dpo_train.py --epochs 3
```

### Custom base model

```bash
python training/dpo_train.py --epochs 3 --base_model mistralai/Mistral-7B-Instruct-v0.3
```

Checkpoints: `checkpoints/dpo-failure-aware/`

---

## 8. Inference & dashboard

### Mock orchestrator (no API keys)

**Windows & Linux**

```bash
python agents/orchestrator.py --query "What is the capital of France?"
```

### FastAPI server + LLM monitoring dashboard

**Windows & Linux**

```bash
python -m app.main
```

| URL | Purpose |
|-----|---------|
| http://localhost:8080/dashboard | **LLM monitoring UI** (live + offline metrics) |
| http://localhost:8080/query | Inference API |
| http://localhost:8080/health | Health check |
| http://localhost:8080/api/monitoring/summary | JSON metrics API |

### Monitoring API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/dashboard` | GET | HTML monitoring UI |
| `/api/monitoring/summary` | GET | Live + offline aggregate stats |
| `/api/monitoring/queries` | GET | Recent query events (`?limit=50`) |
| `/api/monitoring/llm` | GET | Recent LLM calls (`?limit=50`) |
| `/api/monitoring/timeseries` | GET | Latency buckets for charts |
| `/api/monitoring/offline` | GET | Rollout/reward aggregates only |

Events are persisted to `data/monitoring.db` (SQLite).

### Test query (curl)

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'
```

### Google ADK + Gemini (needs `GOOGLE_API_KEY` in `.env`)

```bash
python agents/orchestrator.py --query "What is the capital of France?" --adk
```

---

## 9. Evaluation

Mock-agent eval (CPU-friendly):

```bash
python evaluation/run_eval.py --max-queries 10 --condition all
```

Reports: ECE, hallucination rate, task F1, re-search rate → `results/`

---

## 10. Docker

**Windows & Linux** (Docker Desktop / Docker Engine installed):

```bash
docker compose up api              # FastAPI on :8080
docker compose up mcp-server       # MCP stdio server
docker compose --profile train up training   # DPO smoke train
```

---

## 11. Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `GOOGLE_API_KEY` | For ADK/Gemini | Live ADK inference |
| `SERPAPI_KEY` | Optional | Web search (else DuckDuckGo → mock) |
| `HF_TOKEN` | Optional | Gated HuggingFace models |
| `WANDB_API_KEY` | Optional | Training logs |
| `BASE_MODEL` | Optional | DPO base model (default Mistral-7B) |
| `OLLAMA_HOST` | Optional | Default `http://localhost:11434` |
| `OLLAMA_MODEL` | Optional | Default Ollama tag for your workflow |

Copy from `.env.example` and edit `.env`.

---

## 12. Command cheat sheet (copy-paste)

> Already set up? Jump here. For first-time setup see [Quick start](#quick-start-first-time).

### Windows PowerShell

```powershell
# Activate (after one-time setup)
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path

# Quick smoke test
python scripts/run_offline_pipeline.py --target-size 200 --max-queries 10 --k 4 --synthetic-only

# API + dashboard
python -m app.main
# → http://localhost:8080/dashboard

# CPU — real HF data + mock LLM
python scripts/run_offline_pipeline.py --target-size 500 --max-queries 20 --k 8

# Ollama rollouts
python scripts/generate_rollouts.py --max-queries 5 --k 2 --ollama-model phi3:mini
python scripts/generate_rollouts.py --max-queries 5 --k 2 --ollama-model tinyllama

# Ollama full pipeline (sample data)
python scripts/run_offline_pipeline.py --target-size 50 --max-queries 5 --k 2 --synthetic-only --ollama-model phi3:mini

# GPU DPO
python training/dpo_train.py --epochs 1 --max_samples 10
python training/dpo_train.py --epochs 3

# CLI inference (no server)
python agents/orchestrator.py --query "What is the capital of France?"
```

### Linux / macOS

```bash
# Activate (after one-time setup)
source .venv/bin/activate
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

# Quick smoke test
python scripts/run_offline_pipeline.py --target-size 200 --max-queries 10 --k 4 --synthetic-only

# API + dashboard
python -m app.main
# → http://localhost:8080/dashboard

# CPU — real HF data + mock LLM
python scripts/run_offline_pipeline.py --target-size 500 --max-queries 20 --k 8

# Ollama rollouts
python scripts/generate_rollouts.py --max-queries 5 --k 2 --ollama-model phi3:mini
python scripts/generate_rollouts.py --max-queries 5 --k 2 --ollama-model tinyllama

# Ollama full pipeline (sample data)
python scripts/run_offline_pipeline.py --target-size 50 --max-queries 5 --k 2 --synthetic-only --ollama-model phi3:mini

# GPU DPO
python training/dpo_train.py --epochs 1 --max_samples 10
python training/dpo_train.py --epochs 3

# CLI inference (no server)
python agents/orchestrator.py --query "What is the capital of France?"
```

---

## Optional: shell wrappers

If you prefer shortcuts, `run.ps1` (Windows) and `run.sh` (Linux) wrap the same commands:

```powershell
.\run.ps1 setup                  # one-time venv + deps
.\run.ps1 cpu-mock-pipeline      # smoke test pipeline
.\run.ps1 ollama-phi3-rollouts   # Ollama rollouts
.\run.ps1 inference-api          # start server (dashboard on :8080)
```

```bash
bash run.sh setup
bash run.sh cpu-mock-pipeline
bash run.sh ollama-rollouts phi3:mini
bash run.sh inference-api
```

**This file (`run.md`) is the full reference.** Start with [Quick start](#quick-start-first-time) if you're new to the project.
