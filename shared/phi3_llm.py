"""
Phi-3-mini-4k-instruct local inference backend.

Model: microsoft/Phi-3-mini-4k-instruct  (~7.6 GB fp32 / ~3.8 GB fp16, CPU-friendly)
Requires: pip install transformers torch accelerate

Usage
-----
    from shared.phi3_llm import Phi3LLM

    llm = Phi3LLM()                            # auto-downloads on first run
    response = llm.generate("Who built Rome?")
    print(response)

CLI (rollouts)
--------------
    python scripts/generate_rollouts.py \\
        --phi3 \\
        --max-queries 20 --k 4

    # or via HF model-id directly:
    python scripts/generate_rollouts.py \\
        --model-id microsoft/Phi-3-mini-4k-instruct \\
        --max-queries 20 --k 4

Phi-3 chat template
--------------------
    <|system|>
    {system}<|end|>
    <|user|>
    {user}<|end|>
    <|assistant|>
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# HuggingFace model id — 4k context window variant (lighter than 128k)
PHI3_MODEL_ID = os.environ.get(
    "PHI3_MODEL_ID", "microsoft/Phi-3-mini-4k-instruct"
)

# ReAct system prompt (same format as the rest of the project)
_REACT_SYSTEM = (
    "You are a ReAct search agent. "
    "Respond in this format:\n"
    "Thought: <reasoning>\n"
    "Action: <web_search|wikipedia_lookup|finish>\n"
    "Action Input: <query or entity>\n"
    "After observations, either continue or finish with:\n"
    "Final Answer: <answer>\n"
    "Confidence: <N>% — <reason>"
)


class Phi3LLM:
    """
    Microsoft Phi-3-mini-4k-instruct via HuggingFace transformers pipeline.

    Works on CPU (float32) or GPU (float16 auto-detected).
    Applies the official Phi-3 chat template so the model stays in-distribution.

    Notes
    -----
    * Model size: ~7.6 GB (float32 CPU) / ~3.8 GB (float16 GPU)
    * First run auto-downloads from HuggingFace Hub (~3.8–7.6 GB)
    * Subsequent runs load from cache instantly
    * trust_remote_code=True is required for the Phi-3 tokenizer
    """

    def __init__(
        self,
        model_id: str = PHI3_MODEL_ID,
        device: str = "auto",
        max_new_tokens: int = 256,
        system_prompt: str = _REACT_SYSTEM,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError as exc:
            raise ImportError(
                "Phi3LLM requires 'transformers' and 'torch'.\n"
                "Install with: pip install transformers torch accelerate"
            ) from exc

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt

        _device = "cuda" if (device == "auto" and torch.cuda.is_available()) else "cpu"
        _dtype = torch.float16 if _device == "cuda" else torch.float32

        logger.info("Loading Phi-3 model %s on %s (%s)", model_id, _device, _dtype)
        logger.info("This may take a few minutes on first run (downloading ~3.8–7.6 GB)...")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=_dtype,
            device_map="auto" if _device == "cuda" else None,
            attn_implementation="eager",   # avoids rope_scaling issues on CPU
        )

        self._pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        self._tokenizer = tokenizer
        logger.info("Phi-3 Mini ready.")

    # ------------------------------------------------------------------
    def _build_prompt(self, user_message: str) -> str:
        """
        Apply the official Phi-3 chat template:
            <|system|>\\n{system}<|end|>\\n<|user|>\\n{user}<|end|>\\n<|assistant|>\\n
        Falls back to manual formatting if apply_chat_template is unavailable.
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_message},
        ]
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Manual Phi-3 template fallback
            return (
                f"<|system|>\n{self.system_prompt}<|end|>\n"
                f"<|user|>\n{user_message}<|end|>\n"
                "<|assistant|>\n"
            )

    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        *,
        context: str = "rollout",
        agent: str = "react_rollout",
    ) -> str:
        """
        Generate a response, log to the monitoring store, and return the text.

        Parameters
        ----------
        prompt:      Raw user question / full ReAct context string.
        temperature: Sampling temperature (default 0.7).
        """
        from shared.llm_logging import log_llm_call

        full_prompt = self._build_prompt(prompt)
        t0 = time.perf_counter()
        try:
            outputs = self._pipe(
                full_prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-6),
                return_full_text=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            text: str = outputs[0]["generated_text"].strip()
            latency_ms = (time.perf_counter() - t0) * 1000

            from shared.llm_logging import log_llm_call

            log_llm_call(
                backend="phi3",
                model=self.model_id,
                prompt=full_prompt,
                response=text,
                success=True,
                context=context,
                agent=agent,
                latency_ms=latency_ms,
            )
            logger.debug("Phi-3 [%.0f ms]: %s…", latency_ms, text[:80])
            return text

        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            from shared.llm_logging import log_llm_call

            log_llm_call(
                backend="phi3",
                model=self.model_id,
                prompt=full_prompt,
                response="",
                success=False,
                error=str(exc),
                context=context,
                agent=agent,
                latency_ms=latency_ms,
            )
            logger.error("Phi-3 generation failed: %s", exc)
            raise


# ---------------------------------------------------------------------------
# Convenience factory used by resolve_llm() in generate_rollouts.py
# ---------------------------------------------------------------------------
def build_phi3(
    model_id: str = PHI3_MODEL_ID,
    max_new_tokens: int = 256,
) -> Phi3LLM:
    """Return a ready-to-use Phi3LLM instance."""
    return Phi3LLM(model_id=model_id, max_new_tokens=max_new_tokens)
