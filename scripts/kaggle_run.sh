#!/usr/bin/env bash
set -euo pipefail

# Simple helper to run the Failure-Aware Search Agents service inside
# a Kaggle kernel or any notebook VM. Intended for interactive use.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

LOGFILE="kaggle_fasa.log"
PORT=8080

# Optional: choose backend (mock|ollama|adk)
INFERENCE_BACKEND=${INFERENCE_BACKEND:-mock}
OLLAMA_MODEL=${OLLAMA_MODEL:-phi3:mini}

echo "Repository root: $REPO_ROOT"
echo "Using INFERENCE_BACKEND=$INFERENCE_BACKEND, OLLAMA_MODEL=$OLLAMA_MODEL"

echo "Installing requirements (may take a few minutes)..."
python -m pip install -r requirements.txt --quiet || true

# Start server in background
echo "Starting FastAPI server (logs -> $LOGFILE)"
python -m app.main > "$LOGFILE" 2>&1 &
PID=$!
sleep 3

echo "Server PID: $PID"

echo
# Wait for /health to become available
echo "Waiting for /health..."
for i in 1 2 3 4 5; do
  if curl -sS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "/health is available"
    break
  fi
  echo "retrying..."
  sleep 2
done

echo
# Show health output
echo "/health output:"
curl -s "http://127.0.0.1:$PORT/health" | jq . || true

echo
# Run a sample query
echo "Running sample query..."
curl -s -X POST "http://127.0.0.1:$PORT/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Who invented the telephone?", "mode": "general"}' | jq . || true

echo
# Tail the logs for visibility
echo "Last 40 lines of $LOGFILE"
tail -n 40 "$LOGFILE" || true

echo
echo "To stop the server: kill $PID"