# Grafana Dashboard — Run Guide

This project ships **Prometheus + Grafana** via Docker for historical charts (latency percentiles, policy distribution, LLM backends, CPU/memory trends). The FASA API exposes a `/metrics` endpoint that Prometheus scrapes.

---

## Architecture

```
FASA API (:8080)          Prometheus (:9090)          Grafana (:3000)
     │                           │                          │
     │  GET /metrics             │  scrape every 10s        │  queries Prometheus
     └──────────────────────────►│─────────────────────────►│
                                 │                          │
                                 │                    pre-loaded dashboards
                                 │                    (FASA folder)
```

**Data flow:** Every `POST /query` updates Prometheus counters/histograms. Grafana reads from Prometheus using the auto-provisioned datasource at `http://prometheus:9090`.

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Docker Desktop** (Windows) or **Docker Engine** (Linux/macOS) | Required for Prometheus + Grafana |
| **FASA API on port 8080** | Prometheus scrapes `host.docker.internal:8080/metrics` |
| **Python venv** (if running API locally) | See [run.md](run.md#1-one-time-setup) for setup |

> **Windows:** Docker Desktop must be running. `host.docker.internal` is configured in `docker-compose.yml` so Prometheus inside Docker can reach the API on your host.

---

## Quick start (recommended)

Use **two terminals**: one for the API, one for the observability stack.

### Terminal 1 — Start the FASA API

**Windows (PowerShell)**

```powershell
cd C:\Users\kaila\Desktop\Kaggle Hackathon\Failure-Aware-Search-Agents
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path
python -m app.main
```

**Linux / macOS**

```bash
cd /path/to/Failure-Aware-Search-Agents
source .venv/bin/activate
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
python -m app.main
```

Verify the metrics endpoint:

```bash
curl http://localhost:8080/metrics
```

You should see lines like `fasa_requests_total`, `fasa_request_latency_seconds`, etc.

### Terminal 2 — Start Prometheus + Grafana

From the repo root:

```bash
docker compose up prometheus grafana
```

Run in the background (detached):

```bash
docker compose up -d prometheus grafana
```

### Open the dashboards

| URL | What you get |
|-----|----------------|
| **http://localhost:3000** | Grafana home (default dashboard: FASA main) |
| **http://localhost:3000/d/fasa-main** | **FASA — Failure-Aware Search Agents** (main dashboard) |
| **http://localhost:3000/d/fasa-service-overview** | **FASA — Service Overview** (embedded in the app UI) |
| **http://localhost:9090** | Prometheus UI (targets, query explorer) |
| **http://localhost:8080/** | FASA app with **embedded Grafana iframe** + native charts |
| **http://localhost:8080/dashboard** | Simpler LLM monitoring UI (SQLite-based, no Grafana iframe) |

---

## Grafana login

| Setting | Value |
|---------|-------|
| **URL** | http://localhost:3000 |
| **Admin user** | `admin` |
| **Admin password** | `fasaadmin` |

Anonymous viewing is **enabled** (Viewer role), so you can browse dashboards without logging in. Embedding in the app UI at `http://localhost:8080/` also works without login (`GF_SECURITY_ALLOW_EMBEDDING=true`).

---

## Pre-provisioned dashboards

Dashboards are loaded automatically from `grafana/dashboards/` on startup.

| UID | Title | Purpose |
|-----|-------|---------|
| `fasa-main` | FASA — Failure-Aware Search Agents | Total requests, success rate, confidence, policy breakdown, LLM latency |
| `fasa-service-overview` | FASA — Service Overview | Query rate, p50/p95/p99 latency, CPU/memory, LLM backend pie charts |

Datasource provisioning: `grafana/provisioning/datasources/prometheus.yml` → Prometheus at `http://prometheus:9090`.

---

## Generate data for the charts

Grafana panels stay empty until the API receives traffic. Send a few test queries:

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

Repeat a few times, then refresh Grafana (auto-refresh is **10s** on both dashboards).

---

## Alternative: run everything with Docker

You can run the API, Prometheus, and Grafana together:

```bash
docker compose up api prometheus grafana
```

| Service | Port | Container name |
|---------|------|------------------|
| API | 8080 | (compose project prefix) |
| Prometheus | 9090 | `fasa-prometheus` |
| Grafana | 3000 | `fasa-grafana` |

Prometheus is configured to scrape `host.docker.internal:8080`, which resolves to your machine’s port 8080 (including when the API container publishes that port).

---

## Prometheus metrics reference

Scrape config: `prometheus/prometheus.yml` — job `fasa-api`, path `/metrics`, interval **10s**.

Custom metrics exposed by the API:

| Metric | Type | Description |
|--------|------|-------------|
| `fasa_requests_total` | Counter | Queries by `policy` and `status` |
| `fasa_request_latency_seconds` | Histogram | End-to-end query latency |
| `fasa_confidence_gauge` | Gauge | Rolling average confidence |
| `fasa_llm_calls_total` | Counter | LLM calls by `backend` and `status` |
| `fasa_llm_latency_seconds` | Histogram | LLM generation latency |
| `fasa_error_rate_gauge` | Gauge | Rolling error rate |
| `fasa_retrieval_quality_gauge` | Gauge | Rolling retrieval quality |
| `fasa_active_users_gauge` | Gauge | Active users proxy |
| `system_cpu_percent` | Gauge | CPU utilization |
| `system_memory_percent` | Gauge | Memory utilization |
| `system_uptime_seconds` | Gauge | Process uptime |

Implementation: `app/metrics.py`.

---

## Useful Docker commands

```bash
# Start observability stack (detached)
docker compose up -d prometheus grafana

# View logs
docker compose logs -f grafana
docker compose logs -f prometheus

# Stop
docker compose stop prometheus grafana

# Stop and remove containers (keeps volumes)
docker compose down

# Full reset including Grafana/Prometheus data volumes
docker compose down -v
```

---

## Troubleshooting

### “Grafana Offline” on http://localhost:8080/

The embedded iframe shows this when Grafana is not reachable. Fix:

```bash
docker compose up -d prometheus grafana
```

Wait ~10–20 seconds, then refresh the page. The status should change to **● Connected**.

### Grafana loads but panels show “No data”

1. Confirm the API is running: `curl http://localhost:8080/health`
2. Confirm metrics exist: `curl http://localhost:8080/metrics | findstr fasa` (Windows) or `curl -s http://localhost:8080/metrics | grep fasa` (Linux/macOS)
3. Send test queries (see [Generate data](#generate-data-for-the-charts))
4. In Prometheus, open **Status → Targets**: http://localhost:9090/targets — job `fasa-api` should be **UP**

### Prometheus target `fasa-api` is DOWN

- API must listen on **0.0.0.0:8080** (default in `app/main.py`)
- On Windows, ensure **Docker Desktop** is running
- If the API runs on a different port, update `prometheus/prometheus.yml` target and restart Prometheus:

  ```bash
  docker compose restart prometheus
  ```

### Port already in use

| Port | Service | Fix |
|------|---------|-----|
| 3000 | Grafana | Stop other apps on 3000, or change `"3000:3000"` in `docker-compose.yml` |
| 9090 | Prometheus | Same for 9090 |
| 8080 | FASA API | Stop other processes or set `PORT=8081` and update Prometheus target |

### Change Grafana admin password

Edit `GF_SECURITY_ADMIN_PASSWORD` in `docker-compose.yml`, then:

```bash
docker compose up -d grafana
```

Or change it in the Grafana UI: **Administration → Users → admin**.

---

## Repo files (Grafana-related)

```
grafana/
├── dashboards/
│   ├── fasa_dashboard.json          # uid: fasa-main
│   └── fasa_service_overview.json   # uid: fasa-service-overview
└── provisioning/
    ├── dashboards/dashboards.yml    # auto-load dashboards
    └── datasources/prometheus.yml   # Prometheus datasource

prometheus/
└── prometheus.yml                   # scrape config for fasa-api

docker-compose.yml                 # prometheus + grafana services
app/metrics.py                     # Prometheus metric definitions
app/static/index.html              # embedded Grafana iframe at /
```

---

## Minimal copy-paste checklist

```powershell
# 1. API (PowerShell)
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path
python -m app.main

# 2. Observability (new terminal)
docker compose up -d prometheus grafana

# 3. Open
#    Grafana:  http://localhost:3000/d/fasa-main
#    App UI:   http://localhost:8080/

# 4. Generate traffic
Invoke-RestMethod -Method POST -Uri "http://localhost:8080/query" -ContentType "application/json" -Body '{"query":"What is the capital of France?"}'
```

For the full project (offline pipeline, Ollama, training), see **[run.md](run.md)**.
