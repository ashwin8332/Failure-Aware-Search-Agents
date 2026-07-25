"""
Generate k=8 ReAct-style rollouts per query for DPO preference mining.

Output: data/rollouts/{query_id}.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.rewards import extract_confidence
from shared.search_tools import search_quality_check, web_search, wikipedia_lookup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REACT_SYSTEM = """You are a ReAct search agent. Respond in this format:
Thought: <reasoning>
Action: <web_search|wikipedia_lookup|finish>
Action Input: <query or entity>
After observations, either continue or finish with:
Final Answer: <answer>
Confidence: <N>% — <reason>
"""


class MockLLM:
    """Deterministic mock LLM for CPU-only rollout generation without GPU."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def generate(self, prompt: str, temperature: float = 1.0) -> str:
        """Produce varied trajectories with different confidence/recovery behaviors."""
        import time

        from monitoring.store import get_store

        t0 = time.perf_counter()
        roll = self.rng.random()
        if roll < 0.2:
            text = (
                "Thought: Retrieval looks weak; I should search again.\n"
                "Action: web_search\nAction Input: rephrased query\n"
                "Observation: [mock]\n"
                "Final Answer: Based on limited evidence, the answer may be uncertain.\n"
                "Confidence: 35% — conflicting snippets found."
            )
        elif roll < 0.4:
            text = (
                "Thought: Sources agree.\nAction: finish\n"
                "Final Answer: I cannot determine with certainty from mock data.\n"
                "Confidence: 55% — partial overlap with question."
            )
        elif roll < 0.6:
            text = (
                "Thought: I will answer confidently.\nAction: finish\n"
                "Final Answer: placeholder wrong answer\n"
                "Confidence: 92% — appears correct."
            )
        elif roll < 0.8:
            text = (
                "Thought: Wikipedia may help.\nAction: wikipedia_lookup\n"
                "Action Input: entity\nObservation: [mock]\n"
                "Final Answer: hedged synthesis from sources.\n"
                "Confidence: 48% — mixed evidence."
            )
        else:
            text = (
                "Thought: Evidence is weak; I will hedge and note uncertainty.\n"
                "Action: web_search\nAction Input: verification query\n"
                "Observation: [mock low quality]\n"
                "Final Answer: INSUFFICIENT_EVIDENCE — best effort answer with explicit uncertainty.\n"
                "Confidence: 25% — retrieval quality low, re-search recommended."
            )
        get_store().record_llm_call(
            backend="mock",
            model="MockLLM",
            latency_ms=(time.perf_counter() - t0) * 1000,
            prompt_chars=len(prompt),
            response_chars=len(text),
            success=True,
            context="rollout",
            agent="react_rollout",
        )
        return text


class HFLLM:
    """Optional HuggingFace causal LM backend."""

    def __init__(self, model_id: str, device: str = "auto") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() and device == "auto" else "cpu"
        logger.info("Loading model %s on %s", model_id, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pipe = pipeline(
            "text-generation",
            model=AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            ),
            tokenizer=self.tokenizer,
        )

    def generate(self, prompt: str, temperature: float = 1.0) -> str:
        import time

        from monitoring.store import get_store

        t0 = time.perf_counter()
        try:
            out = self.pipe(
                prompt,
                max_new_tokens=256,
                do_sample=True,
                temperature=temperature,
                return_full_text=False,
            )
            text = out[0]["generated_text"]
            get_store().record_llm_call(
                backend="huggingface",
                model=self.model_id,
                latency_ms=(time.perf_counter() - t0) * 1000,
                prompt_chars=len(prompt),
                response_chars=len(text),
                success=True,
                context="rollout",
                agent="react_rollout",
            )
            return text
        except Exception as exc:
            get_store().record_llm_call(
                backend="huggingface",
                model=self.model_id,
                latency_ms=(time.perf_counter() - t0) * 1000,
                prompt_chars=len(prompt),
                response_chars=0,
                success=False,
                error=str(exc),
                context="rollout",
            )
            raise


def resolve_llm(
    *,
    model_id: Optional[str] = None,
    ollama_model: Optional[str] = None,
    tinyllama: bool = False,
    phi3: bool = False,
    seed: int = 42,
) -> Any:
    if phi3:
        from shared.phi3_llm import build_phi3

        logger.info("Using Phi-3-mini-4k-instruct backend")
        return build_phi3()
    if tinyllama:
        from shared.tinyllama_llm import build_tinyllama

        logger.info("Using TinyLlama-1.1B-Chat backend")
        return build_tinyllama()
    if ollama_model:
        from shared.ollama_llm import OllamaLLM

        return OllamaLLM(ollama_model)
    if model_id:
        return HFLLM(model_id)
    return MockLLM(seed=seed)


class ReActRolloutAgent:
    """ReAct agent that records full trajectories for preference learning."""

    def __init__(
        self,
        llm: Any,
        search_fn: Callable[[str, int], Dict] = web_search,
        wiki_fn: Callable[[str], Dict] = wikipedia_lookup,
        k: int = 8,
        max_turns: int = 5,
        temperature: float = 1.0,
    ) -> None:
        self.llm = llm
        self.search_fn = search_fn
        self.wiki_fn = wiki_fn
        self.k = k
        self.max_turns = max_turns
        self.temperature = temperature

    def _parse_action(self, text: str) -> tuple[str, str, str, Optional[str]]:
        action_m = re.search(r"Action:\s*(\w+)", text, re.I)
        input_m = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text, re.I)
        thought_m = re.search(r"Thought:\s*(.+?)(?:\nAction:|\nFinal Answer:)", text, re.S | re.I)
        final_m = re.search(r"Final Answer:\s*(.+?)(?:\nConfidence:|\Z)", text, re.S | re.I)

        action = action_m.group(1).strip().lower() if action_m else "finish"
        action_input = input_m.group(1).strip() if input_m else ""
        thought = thought_m.group(1).strip() if thought_m else text[:200]
        final = final_m.group(1).strip() if final_m else None
        return thought, action, action_input, final

    def run_single(self, query_row: Dict[str, Any], seed: int) -> Dict[str, Any]:
        question = query_row["question"]
        steps: List[Dict[str, str]] = []
        triggered_research = False
        search_count = 0
        retrieval_qualities: List[float] = []
        context = f"{REACT_SYSTEM}\nQuestion: {question}\n"

        for turn in range(self.max_turns):
            response = self.llm.generate(context, temperature=self.temperature)
            thought, action, action_input, final_answer = self._parse_action(response)

            if action in ("finish", "final") or final_answer:
                if not final_answer:
                    final_answer = action_input or "No answer produced."
                conf = extract_confidence(response)
                avg_rq = sum(retrieval_qualities) / len(retrieval_qualities) if retrieval_qualities else 0.5
                return {
                    "trajectory_id": str(uuid.uuid4()),
                    "query_id": query_row["query_id"],
                    "question": question,
                    "gold_answer": query_row.get("gold_answer", ""),
                    "ambiguity_type": query_row.get("ambiguity_type", ""),
                    "steps": steps,
                    "final_answer": final_answer,
                    "expressed_confidence": conf,
                    "triggered_research": triggered_research,
                    "retrieval_quality": avg_rq,
                }

            observation = ""
            if action == "web_search":
                search_count += 1
                if search_count > 1:
                    triggered_research = True
                result = self.search_fn(action_input or question, 5)
                snippets = [r.get("snippet", "") for r in result.get("results", [])]
                rq = result.get("retrieval_quality", 0.0)
                retrieval_qualities.append(rq)
                quality = search_quality_check(question, snippets)
                observation = json.dumps({
                    "results": result.get("results", [])[:3],
                    "retrieval_quality": rq,
                    "recommendation": quality.get("recommendation"),
                })[:1500]
            elif action == "wikipedia_lookup":
                result = self.wiki_fn(action_input or question)
                observation = json.dumps(result)[:1500]
            else:
                observation = f"Unknown action: {action}"

            steps.append({
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "observation": observation,
            })
            context += f"\nThought: {thought}\nAction: {action}\nAction Input: {action_input}\nObservation: {observation}\n"

        conf = 0.4
        return {
            "trajectory_id": str(uuid.uuid4()),
            "query_id": query_row["query_id"],
            "question": question,
            "gold_answer": query_row.get("gold_answer", ""),
            "ambiguity_type": query_row.get("ambiguity_type", ""),
            "steps": steps,
            "final_answer": "INSUFFICIENT_EVIDENCE: max turns reached.",
            "expressed_confidence": conf,
            "triggered_research": triggered_research,
            "retrieval_quality": sum(retrieval_qualities) / max(1, len(retrieval_qualities)),
        }

    def run_k(self, query_row: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [self.run_single(query_row, seed=i) for i in range(self.k)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate agent rollouts")
    parser.add_argument("--data", default="data/ambiguous_queries.parquet")
    parser.add_argument("--output-dir", default="data/rollouts")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--model-id", default=None, help="HF model; omit for mock LLM")
    parser.add_argument(
        "--ollama-model",
        default=None,
        help="Ollama model tag (e.g. tinyllama, phi3:mini, qwen2.5:3b)",
    )
    parser.add_argument(
        "--tinyllama",
        action="store_true",
        help="Use TinyLlama-1.1B-Chat via HuggingFace transformers (CPU-friendly)",
    )
    parser.add_argument(
        "--phi3",
        action="store_true",
        help="Use Phi-3-mini-4k-instruct via HuggingFace transformers (CPU/GPU)",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data).head(args.max_queries)
    llm: Any = resolve_llm(
        model_id=args.model_id,
        ollama_model=args.ollama_model,
        tinyllama=args.tinyllama,
        phi3=args.phi3,
        seed=args.seed,
    )
    agent = ReActRolloutAgent(llm=llm, k=args.k, temperature=args.temperature)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="rollouts"):
        query = row.to_dict()
        trajectories = agent.run_k(query)
        out_path = Path(args.output_dir) / f"{query['query_id']}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for traj in trajectories:
                f.write(json.dumps(traj) + "\n")

    logger.info("Wrote rollouts for %d queries to %s", len(df), args.output_dir)


if __name__ == "__main__":
    main()
