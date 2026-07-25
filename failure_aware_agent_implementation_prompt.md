# 🧠 IMPLEMENTATION PROMPT: Failure-Aware Search Agents
### Kaggle AI Agents Hackathon 2025 — Full Coding Specification

---

## MASTER CONTEXT

You are implementing **

**, a system that post-trains LLM search agents
using Direct Preference Optimization (DPO) so they express calibrated uncertainty instead of
hallucinating when retrieval results are ambiguous or conflicting.

The system has **two interacting phases**:
1. **Offline DPO Training Pipeline** — generates preference pairs from agent rollouts and fine-tunes the LLM
2. **Inference-Time Multi-Agent System** — Google ADK orchestration with MCP search tools, powered by the DPO-aligned model

**Base model:** `mistralai/Mistral-7B-Instruct-v0.3`
**Framework:** Google ADK + FastMCP + HuggingFace TRL
**Reference repo:** `abhay-lal/failure-aware-search-agents`

---

## PROJECT STRUCTURE TO SCAFFOLD

```
failure-aware-search-agents/
├── data/
│   ├── ambiguous_queries.parquet
│   └── ambiguous_test_set.parquet
├── scripts/
│   ├── generate_rollouts.py
│   ├── score_trajectories.py
│   └── build_preference_pairs.py
├── train/
│   └── dpo_train.py
├── agents/
│   ├── search_agent.py
│   ├── verification_agent.py
│   └── orchestrator.py
├── mcp/
│   └── search_mcp_server.py
├── evaluation/
│   └── run_eval.py
├── prompts/
│   ├── search_agent_prompt.txt
│   └── verify_agent_prompt.txt
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## PHASE 1 — DATASET CONSTRUCTION

### Task: Build `scripts/generate_dataset.py`

Load and merge three HuggingFace QA datasets. Tag each query with an ambiguity type.
Output: `data/ambiguous_queries.parquet`

```
Requirements:
- Load: hotpot_qa (distractor split), trivia_qa (unfiltered.nocontext), ambig_qa
- Normalize all entries to schema: {query_id, question, gold_answer, source, ambiguity_type}
- Assign ambiguity_type from: ["temporal", "conflicting_sources", "multi_hop",
  "out_of_scope", "near_miss"]
- Use a heuristic to assign ambiguity_type per source:
    hotpot_qa → "multi_hop"
    trivia_qa → "near_miss"
    ambig_qa  → "conflicting_sources"
- Target size: 5,000–10,000 rows, balanced across types (use stratified sampling)
- Save as parquet with columns: query_id, question, gold_answer, source, ambiguity_type
- Add a train/test split flag (90/10)
```

**Key imports:** `datasets`, `pandas`, `uuid`, `sklearn.model_selection`

---

## PHASE 2 — AGENT ROLLOUT GENERATOR

### Task: Build `scripts/generate_rollouts.py`

Implement a `ReActRolloutAgent` class that runs k=8 trajectories per query using a
base Mistral-7B model with ReAct prompting.

```
Requirements:
- Class: ReActRolloutAgent(model_id, search_tools, k=8)
- Each trajectory = full Thought → Action → Observation → Answer loop
- Max turns per trajectory: 5
- Temperature: 1.0 (for diversity)
- Record per trajectory:
    trajectory_id (uuid)
    query_id
    steps: List[{thought, action, action_input, observation}]
    final_answer: str
    expressed_confidence: float  (extract from text: "I am X% confident...")
    triggered_research: bool     (did the agent re-search mid-trajectory?)
    retrieval_quality: float     (score of the observations returned)
- Output: trajectories saved as JSON lines to data/rollouts/{query_id}.jsonl
- Use temperature sampling; each of 8 rollouts should diverge

Search tools to mock/stub if no API key:
    web_search(query) → returns list of {title, snippet, url, quality_score}
    wikipedia_lookup(entity) → returns {summary, url}

Confidence extraction regex:
    r"(?:I am|confidence:|certainty:)\s*(\d{1,3})%"
    If not found, default expressed_confidence = 0.75

Retrieval quality heuristic:
    Check if snippets are relevant using keyword overlap with question
    Score = (matching_keywords / total_keywords).clip(0, 1)
```

**Key imports:** `transformers`, `torch`, `json`, `re`, `uuid`, `tqdm`

---

## PHASE 3 — MULTI-COMPONENT REWARD SCORER

### Task: Build `scripts/score_trajectories.py`

Score each trajectory on 3 dimensions and compute composite reward.

```
Composite reward formula:
    R(τ) = 0.5 * R_fact(τ) + 0.3 * R_calib(τ) + 0.2 * R_recovery(τ)

Component definitions:

R_fact — Factuality Score
    Use BERTScore(traj.final_answer, gold_answer, lang="en")
    Return F1 score as float in [0, 1]
    Fallback: use exact_match + token_f1 if BERTScore unavailable

R_calib — Calibration Score
    expressed_conf = traj.expressed_confidence  # float in [0,1]
    is_correct = exact_match(traj.final_answer, gold_answer)
    R_calib = 1 - abs(expressed_conf - float(is_correct))
    Interpretation: penalizes HIGH conf on WRONG answers and LOW conf on CORRECT answers

R_recovery — Failure Recovery Score
    retrieval_quality = traj.retrieval_quality  # float in [0,1]
    R_recovery = float(traj.triggered_research) * (1 - retrieval_quality)
    Interpretation: bonus for re-searching when retrieval was poor, zero if retrieval was good

Implementation:
    def score_trajectory(traj: dict, gold_answer: str) -> float

Also output a per-trajectory scored JSONL: data/scored_trajectories.jsonl
Add fields: r_fact, r_calib, r_recovery, composite_reward to each trajectory record
```

**Key imports:** `bert_score`, `evaluate`, `json`, `pathlib`

---

## PHASE 4 — PREFERENCE PAIR BUILDER

### Task: Build `scripts/build_preference_pairs.py`

Group trajectories by query_id, select best (chosen) and worst (rejected),
filter by reward margin, format for DPOTrainer.

```
Algorithm:
    FOR each query_id:
        Sort trajectories by composite_reward descending
        chosen = trajectories[0]   # highest reward
        rejected = trajectories[-1] # lowest reward
        reward_gap = chosen.reward - rejected.reward
        IF reward_gap > 0.3:
            keep pair
        ELSE:
            discard (not discriminative enough)

Output schema (HuggingFace Dataset format):
    {
      "prompt": "<query text>",
      "chosen": "<full trajectory text of best rollout>",
      "rejected": "<full trajectory text of worst rollout>",
      "reward_gap": float,
      "ambiguity_type": str,
      "query_id": str
    }

Trajectory text format:
    "Question: {question}\n\n
     Thought: {step.thought}\n
     Action: {step.action}[{step.action_input}]\n
     Observation: {step.observation}\n
     ... (repeat for each step)
     Final Answer: {final_answer}\nConfidence: {confidence}%"

Save as: data/preference_pairs/
    train_pairs.jsonl
    val_pairs.jsonl  (10% holdout)
    summary_stats.json  (pair counts per ambiguity_type, mean reward_gap)

Log and print:
    Total queries processed
    Pairs kept vs discarded
    Mean reward gap of kept pairs
    Distribution by ambiguity_type
```

**Key imports:** `datasets`, `json`, `collections`, `pandas`, `sklearn`

---

## PHASE 5 — DPO FINE-TUNING

### Task: Build `train/dpo_train.py`

Fine-tune Mistral-7B-Instruct-v0.3 with QLoRA + TRL's DPOTrainer.

```
Model loading:
    - 4-bit NF4 quantization via BitsAndBytesConfig
    - bnb_4bit_compute_dtype = torch.bfloat16
    - device_map = "auto"

LoRA config:
    r = 64
    lora_alpha = 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_dropout = 0.05
    bias = "none"
    task_type = "CAUSAL_LM"

DPO hyperparameters:
    beta = 0.1                      # KL regularization from reference
    learning_rate = 5e-5
    num_train_epochs = 3
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 4
    max_length = 1024
    max_prompt_length = 512
    output_dir = "./checkpoints/dpo-failure-aware"
    logging_steps = 10
    save_steps = 100
    eval_steps = 100
    warmup_ratio = 0.1
    lr_scheduler_type = "cosine"
    optim = "paged_adamw_32bit"
    fp16 = False
    bf16 = True

Reference model:
    Load a frozen copy of the same base model (NOT the LoRA version)
    ref_model = AutoModelForCausalLM.from_pretrained(base_model_id, ...)

Weights & Biases logging:
    wandb.init(project="failure-aware-dpo", config={...hyperparams})
    Log: train/loss, eval/loss, train/rewards/chosen, train/rewards/rejected,
         train/rewards/margins, train/logps/chosen, train/logps/rejected

Dataset loading:
    Load data/preference_pairs/train_pairs.jsonl
    Load data/preference_pairs/val_pairs.jsonl
    Use datasets.Dataset.from_json(...)

After training:
    Save merged model: trainer.model.merge_and_unload()
    Save tokenizer
    Push optional summary metrics to wandb

Add CLI args via argparse:
    --base_model (default: mistralai/Mistral-7B-Instruct-v0.3)
    --data_dir   (default: data/preference_pairs)
    --output_dir (default: checkpoints/dpo-failure-aware)
    --beta       (default: 0.1)
    --epochs     (default: 3)
    --wandb_project (default: failure-aware-dpo)
```

**Key imports:** `trl`, `peft`, `transformers`, `bitsandbytes`, `torch`, `wandb`, `datasets`, `argparse`

---

## PHASE 6 — MCP SEARCH SERVER

### Task: Build `mcp/search_mcp_server.py`

A FastMCP server exposing 4 tools: web_search, wikipedia_lookup,
coherence_score, and search_quality_check.

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("failure-aware-search")

@mcp.tool()
async def web_search(query: str, num_results: int = 5) -> dict:
    """
    Search the web via SerpAPI. Returns results with quality metadata.
    Args:
        query: search query string
        num_results: number of results to return (max 10)
    Returns:
        {
          "results": List[{title, snippet, url, position}],
          "coherence_score": float,    # avg pairwise semantic similarity of snippets
          "conflict_detected": bool,   # True if snippets contradict each other
          "retrieval_quality": float   # overall quality score [0,1]
        }
    """

@mcp.tool()
async def wikipedia_lookup(entity: str) -> dict:
    """
    Fetch Wikipedia summary for authoritative grounding.
    Returns: {summary: str, url: str, categories: List[str], found: bool}
    """

@mcp.tool()
async def coherence_score(texts: List[str]) -> dict:
    """
    Score semantic coherence of a list of text passages.
    Returns: {score: float, conflict_pairs: List[Tuple[int,int]]}
    """

@mcp.tool()
async def search_quality_check(query: str, snippets: List[str]) -> dict:
    """
    Assess retrieval quality: relevance + recency + source diversity.
    Returns: {quality_score: float, reasons: List[str], recommendation: str}
    recommendation ∈ ["confident", "hedge", "re_search"]
    """

Security requirements:
    - Load SERPAPI_KEY from environment (os.environ.get), NEVER hardcode
    - Validate and sanitize all incoming query strings (strip, max 500 chars)
    - Rate limit: max 10 calls/minute per tool using a token bucket
    - If SERPAPI_KEY missing, fall back to DuckDuckGo (duckduckgo_search library)

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

---

## PHASE 7 — GOOGLE ADK MULTI-AGENT SYSTEM

### Task: Build `agents/orchestrator.py`, `agents/search_agent.py`, `agents/verification_agent.py`

#### `agents/search_agent.py`
```
LlmAgent config:
    model = 'failure-aware-mistral-7b'  (or 'gemini-1.5-flash' for demo)
    name = 'search_agent'
    tools = [web_search, wikipedia_lookup, search_quality_check]  (from MCP)
    instruction = load_prompt("prompts/search_agent_prompt.txt")

Instruction prompt must tell the agent to:
    1. Use web_search for the user query
    2. Check search_quality_check result
    3. IF quality < 0.5 → trigger re-search with rephrased query
    4. Express confidence level explicitly: "I am X% confident because..."
    5. If conflicting info found → say "Sources conflict on this. Best available answer: ..."
    6. Max 3 search attempts before hedging with "Insufficient evidence found"
```

#### `agents/verification_agent.py`
```
LlmAgent config:
    model = 'failure-aware-mistral-7b'
    name = 'verification_agent'
    tools = [wikipedia_lookup, coherence_score]
    instruction = load_prompt("prompts/verify_agent_prompt.txt")

Instruction prompt must tell the agent to:
    1. Cross-check the search_agent's answer against Wikipedia
    2. Run coherence_score on the combined snippets
    3. If conflict detected → flag it explicitly
    4. Output: {verdict: "confirmed" | "conflicted" | "unverified", confidence_adjustment: float}
```

#### `agents/orchestrator.py`
```python
from google.adk.agents import SequentialAgent, LlmAgent
from google.adk.tools.mcp_tool import MCPToolset, StdioServerParameters

async def build_agent():
    search_tools, exit_stack = await MCPToolset.from_server(
        connection_params=StdioServerParameters(
            command='python',
            args=['mcp/search_mcp_server.py']
        )
    )

    search_agent = build_search_agent(search_tools)
    verify_agent  = build_verification_agent(search_tools)

    orchestrator = SequentialAgent(
        name='failure_aware_orchestrator',
        sub_agents=[search_agent, verify_agent],
        description="Routes query to search + verification agents"
    )
    return orchestrator, exit_stack

# Entry point
if __name__ == "__main__":
    import asyncio
    from google.adk.runners import Runner
    asyncio.run(Runner(build_agent).run())
```

---

## PHASE 8 — SYSTEM PROMPTS

### Task: Write `prompts/search_agent_prompt.txt`

```
You are a failure-aware search agent. Your primary duty is epistemic honesty.

When given a query:
1. Search for evidence using web_search.
2. Immediately run search_quality_check on the results.
3. If quality_score < 0.5 OR conflict_detected == true:
   - Rephrase the query and search again (max 3 attempts total).
   - If quality is still poor after 3 attempts, respond with:
     "INSUFFICIENT_EVIDENCE: I searched 3 times but found conflicting or
     low-quality results. Best available: [answer]. Confidence: LOW (<40%)."
4. Always state your confidence explicitly:
   "I am [N]% confident. Reason: [brief justification]."
5. Never fabricate citations or invent facts.
6. Prefer saying "I don't know" over a confident wrong answer.

Output format:
Thought: [your reasoning]
Action: [tool name]
Action Input: [tool input]
Observation: [tool result]
... (repeat up to 5 turns)
Final Answer: [answer]
Confidence: [N]% — [reason]
```

### Task: Write `prompts/verify_agent_prompt.txt`

```
You are a verification agent. Given a draft answer from the search agent:
1. Cross-check the core claim via wikipedia_lookup.
2. Run coherence_score on [search snippets + wikipedia summary].
3. Output ONLY a JSON verdict:
{
  "verdict": "confirmed" | "conflicted" | "unverified",
  "confidence_adjustment": float (-0.3 to +0.1),
  "conflict_summary": "..." or null,
  "verified_facts": ["...", "..."]
}
Do not add conversational filler. JSON only.
```

---

## PHASE 9 — EVALUATION

### Task: Build `evaluation/run_eval.py`

```
Evaluate two agents — baseline (Mistral-7B no DPO) vs DPO-aligned agent.

Metrics to compute:
    1. ECE (Expected Calibration Error)
       - Bin expressed_confidence into 10 equal-width buckets
       - ECE = sum_b (|B_b|/N) * |accuracy_b - confidence_b|
       - Target: < 0.05

    2. Hallucination Rate
       - Definition: (wrong_answer AND high_confidence > 0.7) / total_queries
       - Target: < 20%

    3. Task Success F1 (AmbigNQ)
       - token_f1 between final_answer and gold_answer
       - Target: > 0.72

    4. Re-search Rate
       - Definition: triggered_research / (queries where retrieval_quality < 0.5)
       - Target: > 65%

    5. Overconfidence Rate
       - Definition: (confidence > 0.7 AND is_wrong) / total_queries
       - Target: < 12%

Output:
    - Print comparison table to console
    - Save results/eval_results.json
    - Optionally log to wandb

Ablation conditions to support via --condition flag:
    baseline      — Mistral-7B-Instruct, no DPO
    sft_only      — SFT on chosen trajectories, no DPO objective
    dpo_fact_only — DPO with reward = R_fact only
    dpo_full      — DPO with full composite reward (R_fact + R_calib + R_recovery)
    dpo_full_adk  — Full method + ADK multi-agent orchestration
```

---

## PHASE 10 — DOCKER & DEPLOYMENT

### Task: Write `Dockerfile` and `docker-compose.yml`

```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["sh", "-c", "python mcp/search_mcp_server.py & adk web --port 8080"]
```

```yaml
# docker-compose.yml
version: '3.9'
services:
  mcp-server:
    build: .
    command: python mcp/search_mcp_server.py
    environment:
      - SERPAPI_KEY=${SERPAPI_KEY}
    restart: unless-stopped

  adk-agent:
    build: .
    command: adk web --port 8080
    ports:
      - "8080:8080"
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - SERPAPI_KEY=${SERPAPI_KEY}
    depends_on:
      - mcp-server
    restart: unless-stopped
```

```
# .env.example
SERPAPI_KEY=your_serpapi_key_here
GOOGLE_API_KEY=your_google_api_key_here
WANDB_API_KEY=your_wandb_key_here
HF_TOKEN=your_huggingface_token_here
# NEVER commit .env to git — it is in .gitignore
```

---

## REQUIREMENTS.TXT

```
# Core
torch>=2.1.0
transformers>=4.40.0
datasets>=2.18.0
tokenizers>=0.19.0

# Training
trl>=0.8.6
peft>=0.10.0
bitsandbytes>=0.43.0
accelerate>=0.29.0
wandb>=0.16.0

# MCP + ADK
google-adk>=0.4.0
mcp>=1.0.0
fastmcp>=0.4.0

# Search
duckduckgo-search>=5.0.0
httpx>=0.27.0
wikipedia-api>=0.6.0

# Evaluation
bert-score>=0.3.13
evaluate>=0.4.1
scikit-learn>=1.4.0
pandas>=2.2.0
numpy>=1.26.0

# Infra
python-dotenv>=1.0.0
tqdm>=4.66.0
pyarrow>=15.0.0
```

---

## CODING RULES TO FOLLOW THROUGHOUT

```
1. Security: Load ALL API keys via os.environ.get("KEY_NAME"). Raise ValueError if missing.
   Never hardcode secrets anywhere.

2. Typing: Use Python type hints on all function signatures.

3. Async: MCP tools must be async def. ADK agents use asyncio.

4. Error handling: Wrap all external API calls in try/except. On failure, return
   degraded but safe output (e.g., {"error": str(e), "retrieval_quality": 0.0}).

5. Logging: Use Python logging module (not print) in production paths.
   Use print() only in eval/reporting scripts.

6. Config: Use argparse for all scripts. No magic hardcoded values — pass as args.

7. Reproducibility: Set random seeds: random.seed(42), np.random.seed(42),
   torch.manual_seed(42) at top of training scripts.

8. GPU fallback: All torch code must handle both CUDA and CPU gracefully.
   Use device = "cuda" if torch.cuda.is_available() else "cpu".

9. Checkpointing: DPO trainer saves checkpoints every 100 steps. Eval resumes
   from checkpoint if interrupted.

10. Tests: For each module, add a simple if __name__ == "__main__" smoke test
    with 3–5 sample queries that runs end-to-end without external APIs (mock tools).
```

---

## SUGGESTED BUILD ORDER

```
Step 1:  requirements.txt + .env.example + project scaffold
Step 2:  scripts/generate_dataset.py  (no GPU needed)
Step 3:  mcp/search_mcp_server.py     (test with mock results first)
Step 4:  scripts/generate_rollouts.py (stub search tools, test trajectory format)
Step 5:  scripts/score_trajectories.py (verify reward math on toy examples)
Step 6:  scripts/build_preference_pairs.py (check pair counts and reward gap dist.)
Step 7:  train/dpo_train.py            (requires GPU; test with tiny dataset first)
Step 8:  agents/ (search_agent, verification_agent, orchestrator)
Step 9:  evaluation/run_eval.py        (test on 50 queries before full eval)
Step 10: Dockerfile + docker-compose.yml
Step 11: README.md with architecture diagram ASCII art
```

---

*Built for Kaggle AI Agents Hackathon 2025 — Failure-Aware Search Agents*
*Stack: Google ADK · FastMCP · HuggingFace TRL · DPO · Mistral-7B · QLoRA*
