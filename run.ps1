# Failure-Aware Search Agents — run commands (Windows PowerShell)
# Usage: .\run.ps1 <command>
#   .\run.ps1 setup
#   .\run.ps1 cpu-mock-pipeline
#   .\run.ps1 ollama-phi3-rollouts

param(
    [Parameter(Position = 0)]
    [string]$Command = "help",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

# Use py launcher on Windows if python is not on PATH
if (Get-Command python -ErrorAction SilentlyContinue) {
    $script:Py = "python"
} else {
    $script:Py = "py"
}

$env:PYTHONPATH = if ($env:PYTHONPATH) { "$Root;$($env:PYTHONPATH)" } else { $Root }

# Ollama models (you have tinyllama + phi3:mini; pull more as needed)
$OllamaTiny = "tinyllama"
$OllamaPhi3 = "phi3:mini"
$OllamaQwenSmall = "qwen2.5:3b"
$OllamaQwenMed = "qwen2.5:7b"

function Activate-Venv {
    $activate = Join-Path $Root ".venv\Scripts\Activate.ps1"
    if (-not (Test-Path $activate)) {
        Write-Host "Run: .\run.ps1 setup"
        exit 1
    }
    . $activate
}

function Invoke-Python {
    param([string[]]$Args)
    & $script:Py @Args
}

function cmd_setup {
    Invoke-Python -Args @("-m", "venv", ".venv")
    . (Join-Path $Root ".venv\Scripts\Activate.ps1")
    Invoke-Python -Args @("-m", "pip", "install", "--upgrade", "pip")
    Invoke-Python -Args @("-m", "pip", "install", "-r", "requirements.txt")
    if (-not (Test-Path ".env")) {
        Copy-Item ".env.example" ".env"
        Write-Host "Created .env from .env.example"
    }
    Write-Host "Setup complete. Activate with: .\.venv\Scripts\Activate.ps1"
}

function cmd_ollama_pull {
    ollama pull $OllamaTiny
    ollama pull $OllamaPhi3
    Write-Host "Optional: ollama pull $OllamaQwenSmall"
    Write-Host "Optional: ollama pull $OllamaQwenMed"
}

function cmd_cpu_mock_pipeline {
    Activate-Venv
    Invoke-Python -Args @(
        "scripts/run_offline_pipeline.py",
        "--target-size", "200", "--max-queries", "10", "--k", "4", "--synthetic-only"
    )
}

function cmd_cpu_mock_dataset {
    Activate-Venv
    Invoke-Python -Args @("scripts/generate_dataset.py", "--target-size", "200", "--synthetic-only")
}

function cmd_cpu_mock_rollouts {
    Activate-Venv
    Invoke-Python -Args @("scripts/generate_rollouts.py", "--max-queries", "10", "--k", "4")
}

function cmd_cpu_mock_score {
    Activate-Venv
    Invoke-Python -Args @("scripts/score_trajectories.py")
    Invoke-Python -Args @("scripts/build_preference_pairs.py")
}

function cmd_cpu_real_dataset {
    Activate-Venv
    Invoke-Python -Args @("scripts/generate_dataset.py", "--target-size", "500")
}

function cmd_cpu_real_pipeline_mock_llm {
    Activate-Venv
    Invoke-Python -Args @(
        "scripts/run_offline_pipeline.py",
        "--target-size", "500", "--max-queries", "20", "--k", "8"
    )
}

function cmd_ollama_rollouts {
    param([string]$Model = $OllamaPhi3)
    Activate-Venv
    Invoke-Python -Args @(
        "scripts/generate_rollouts.py",
        "--max-queries", "5", "--k", "2", "--ollama-model", $Model
    )
}

function cmd_ollama_pipeline_sample {
    param([string]$Model = $OllamaPhi3)
    Activate-Venv
    Invoke-Python -Args @(
        "scripts/run_offline_pipeline.py",
        "--target-size", "50", "--max-queries", "5", "--k", "2",
        "--synthetic-only", "--ollama-model", $Model
    )
}

function cmd_gpu_dpo_smoke {
    Activate-Venv
    Invoke-Python -Args @("training/dpo_train.py", "--epochs", "1", "--max_samples", "10")
}

function cmd_gpu_dpo_full {
    Activate-Venv
    Invoke-Python -Args @("training/dpo_train.py", "--epochs", "3")
}

function cmd_inference_mock {
    Activate-Venv
    Invoke-Python -Args @(
        "agents/orchestrator.py",
        "--query", "What is the capital of France?"
    )
}

function cmd_inference_api {
    Activate-Venv
    Invoke-Python -Args @("-m", "app.main")
}

function cmd_inference_adk {
    Activate-Venv
    Invoke-Python -Args @(
        "agents/orchestrator.py",
        "--query", "What is the capital of France?", "--adk"
    )
}

function cmd_eval_mock {
    Activate-Venv
    Invoke-Python -Args @("evaluation/run_eval.py", "--max-queries", "10", "--condition", "all")
}

function cmd_help {
    @"
Failure-Aware Search Agents — run.ps1 commands

Setup
  setup                     Create venv, pip install, copy .env
  ollama-pull               Pull tinyllama + phi3:mini

CPU + sample data + mock LLM
  cpu-mock-pipeline         Full offline pipeline (synthetic, mock LLM)
  cpu-mock-dataset          Synthetic dataset only
  cpu-mock-rollouts         Mock LLM rollouts only
  cpu-mock-score            Score + preference pairs

CPU + real HF datasets
  cpu-real-dataset          Download HotpotQA / TriviaQA / AmbigQA
  cpu-real-pipeline-mock-llm  Real data, mock LLM rollouts

Ollama
  ollama-rollouts [model]   Rollouts (default: phi3:mini)
  ollama-pipeline-sample [model]  Synthetic pipeline + Ollama
  ollama-phi3-rollouts        phi3:mini rollouts
  ollama-tiny-rollouts        tinyllama rollouts

GPU DPO
  gpu-dpo-smoke             1 epoch, 10 samples
  gpu-dpo-full              3 epochs

Inference
  inference-mock            Mock orchestrator
  inference-api             FastAPI :8080
  inference-adk             Google ADK

Eval
  eval-mock                 Evaluation with mock agent

Env: OLLAMA_HOST=http://localhost:11434  OLLAMA_MODEL=phi3:mini
"@ | Write-Host
}

$extra = if ($Rest.Count -gt 0) { $Rest[0] } else { $null }

switch ($Command) {
    "setup" { cmd_setup }
    "ollama-pull" { cmd_ollama_pull }
    "cpu-mock-pipeline" { cmd_cpu_mock_pipeline }
    "cpu-mock-dataset" { cmd_cpu_mock_dataset }
    "cpu-mock-rollouts" { cmd_cpu_mock_rollouts }
    "cpu-mock-score" { cmd_cpu_mock_score }
    "cpu-real-dataset" { cmd_cpu_real_dataset }
    "cpu-real-pipeline-mock-llm" { cmd_cpu_real_pipeline_mock_llm }
    "ollama-rollouts" { cmd_ollama_rollouts -Model $(if ($extra) { $extra } else { $OllamaPhi3 }) }
    "ollama-pipeline-sample" { cmd_ollama_pipeline_sample -Model $(if ($extra) { $extra } else { $OllamaPhi3 }) }
    "ollama-phi3-rollouts" { cmd_ollama_rollouts -Model $OllamaPhi3 }
    "ollama-tiny-rollouts" { cmd_ollama_rollouts -Model $OllamaTiny }
    "gpu-dpo-smoke" { cmd_gpu_dpo_smoke }
    "gpu-dpo-full" { cmd_gpu_dpo_full }
    "inference-mock" { cmd_inference_mock }
    "inference-api" { cmd_inference_api }
    "inference-adk" { cmd_inference_adk }
    "eval-mock" { cmd_eval_mock }
    "help" { cmd_help }
    default {
        Write-Host "Unknown command: $Command"
        cmd_help
        exit 1
    }
}
