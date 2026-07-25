#!/usr/bin/env bash
# Failure-Aware Search Agents — run commands (Linux / macOS)
# Usage: bash run.sh <command>
#   bash run.sh setup
#   bash run.sh cpu-mock-pipeline
#   bash run.sh ollama-phi3-rollouts
#   bash run.sh help

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PY:-python3}"
VENV_ACTIVATE="source .venv/bin/activate"

# ---------------------------------------------------------------------------
# Ollama models (install with: ollama pull <name>)
# You have: tinyllama, phi3:mini — add more as needed:
#   ollama pull qwen2.5:3b
#   ollama pull qwen2.5:7b
#   ollama pull mistral:7b
# ---------------------------------------------------------------------------
OLLAMA_TINY="tinyllama"
OLLAMA_PHI3="phi3:mini"
OLLAMA_QWEN_SMALL="qwen2.5:3b"
OLLAMA_QWEN_MED="qwen2.5:7b"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

activate_venv() {
  if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
  else
    echo "Run: bash run.sh setup"
    exit 1
  fi
}

cmd_setup() {
  $PY -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "Created .env from .env.example"
  fi
  echo "Setup complete. Activate with: source .venv/bin/activate"
}

cmd_ollama_pull() {
  ollama pull "$OLLAMA_TINY"
  ollama pull "$OLLAMA_PHI3"
  echo "Optional larger models:"
  echo "  ollama pull $OLLAMA_QWEN_SMALL"
  echo "  ollama pull $OLLAMA_QWEN_MED"
}

# --- CPU + sample/synthetic data + mock LLM (fastest, no GPU, no Ollama) ---
cmd_cpu_mock_pipeline() {
  activate_venv
  python scripts/run_offline_pipeline.py \
    --target-size 200 --max-queries 10 --k 4 --synthetic-only
}

cmd_cpu_mock_dataset() {
  activate_venv
  python scripts/generate_dataset.py --target-size 200 --synthetic-only
}

cmd_cpu_mock_rollouts() {
  activate_venv
  python scripts/generate_rollouts.py --max-queries 10 --k 4
}

cmd_cpu_mock_score() {
  activate_venv
  python scripts/score_trajectories.py
  python scripts/build_preference_pairs.py
}

# --- CPU + real HuggingFace datasets (downloads; needs network) ---
cmd_cpu_real_dataset() {
  activate_venv
  python scripts/generate_dataset.py --target-size 500
}

cmd_cpu_real_pipeline_mock_llm() {
  activate_venv
  python scripts/run_offline_pipeline.py --target-size 500 --max-queries 20 --k 8
}

# --- Ollama rollouts (CPU/GPU handled by Ollama; sample or real data) ---
cmd_ollama_rollouts() {
  local model="${1:-$OLLAMA_PHI3}"
  activate_venv
  python scripts/generate_rollouts.py \
    --max-queries 5 --k 2 --ollama-model "$model"
}

cmd_ollama_pipeline_sample() {
  local model="${1:-$OLLAMA_PHI3}"
  activate_venv
  python scripts/run_offline_pipeline.py \
    --target-size 50 --max-queries 5 --k 2 \
    --synthetic-only --ollama-model "$model"
}

cmd_ollama_phi3_rollouts() { cmd_ollama_rollouts "$OLLAMA_PHI3"; }
cmd_ollama_tiny_rollouts() { cmd_ollama_rollouts "$OLLAMA_TINY"; }

# --- GPU DPO training (CUDA recommended; smoke vs full) ---
cmd_gpu_dpo_smoke() {
  activate_venv
  python training/dpo_train.py --epochs 1 --max_samples 10
}

cmd_gpu_dpo_full() {
  activate_venv
  python training/dpo_train.py --epochs 3
}

# --- Inference (mock orchestrator / API / ADK) ---
cmd_inference_mock() {
  activate_venv
  python agents/orchestrator.py --query "What is the capital of France?"
}

cmd_inference_api() {
  activate_venv
  python -m app.main
  # POST http://localhost:8080/query  {"query": "..."}
}

cmd_inference_adk() {
  activate_venv
  python agents/orchestrator.py --query "What is the capital of France?" --adk
}

# --- Evaluation ---
cmd_eval_mock() {
  activate_venv
  python evaluation/run_eval.py --max-queries 10 --condition all
}

# --- Docker ---
cmd_docker_api() {
  docker compose up api
}

cmd_docker_mcp() {
  docker compose up mcp-server
}

cmd_docker_train_smoke() {
  docker compose --profile train up training
}

cmd_help() {
  cat <<'EOF'
Failure-Aware Search Agents — run.sh commands

Setup
  setup                     Create venv, pip install, copy .env
  ollama-pull               Pull tinyllama + phi3:mini

CPU + sample data + mock LLM (no Ollama, no GPU)
  cpu-mock-pipeline         Full offline pipeline (synthetic data, mock LLM)
  cpu-mock-dataset          Build synthetic dataset only
  cpu-mock-rollouts         Mock LLM rollouts only
  cpu-mock-score            Score + preference pairs

CPU + real HF datasets
  cpu-real-dataset          Download HotpotQA / TriviaQA / AmbigQA
  cpu-real-pipeline-mock-llm  Real data pipeline, mock LLM rollouts

Ollama (local models — CPU/GPU via Ollama)
  ollama-rollouts [model]   Rollouts with Ollama (default: phi3:mini)
  ollama-pipeline-sample [model]  Synthetic pipeline + Ollama rollouts
  ollama-phi3-rollouts      Shorthand for phi3:mini
  ollama-tiny-rollouts      Shorthand for tinyllama

GPU DPO training
  gpu-dpo-smoke             1 epoch, 10 samples
  gpu-dpo-full              3 epochs full run

Inference
  inference-mock            Mock orchestrator CLI
  inference-api             FastAPI on :8080
  inference-adk             Google ADK (needs GOOGLE_API_KEY)

Eval & Docker
  eval-mock                 Evaluation with mock agent
  docker-api / docker-mcp / docker-train-smoke

Env overrides: PY=python3 OLLAMA_HOST=http://localhost:11434
EOF
}

main() {
  local cmd="${1:-help}"
  shift || true
  case "$cmd" in
    setup) cmd_setup ;;
    ollama-pull) cmd_ollama_pull ;;
    cpu-mock-pipeline) cmd_cpu_mock_pipeline ;;
    cpu-mock-dataset) cmd_cpu_mock_dataset ;;
    cpu-mock-rollouts) cmd_cpu_mock_rollouts ;;
    cpu-mock-score) cmd_cpu_mock_score ;;
    cpu-real-dataset) cmd_cpu_real_dataset ;;
    cpu-real-pipeline-mock-llm) cmd_cpu_real_pipeline_mock_llm ;;
    ollama-rollouts) cmd_ollama_rollouts "$@" ;;
    ollama-pipeline-sample) cmd_ollama_pipeline_sample "$@" ;;
    ollama-phi3-rollouts) cmd_ollama_phi3_rollouts ;;
    ollama-tiny-rollouts) cmd_ollama_tiny_rollouts ;;
    gpu-dpo-smoke) cmd_gpu_dpo_smoke ;;
    gpu-dpo-full) cmd_gpu_dpo_full ;;
    inference-mock) cmd_inference_mock ;;
    inference-api) cmd_inference_api ;;
    inference-adk) cmd_inference_adk ;;
    eval-mock) cmd_eval_mock ;;
    docker-api) cmd_docker_api ;;
    docker-mcp) cmd_docker_mcp ;;
    docker-train-smoke) cmd_docker_train_smoke ;;
    help|-h|--help) cmd_help ;;
    *) echo "Unknown command: $cmd"; cmd_help; exit 1 ;;
  esac
}

main "$@"
