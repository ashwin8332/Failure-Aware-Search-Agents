"""
TinyLlama-1.1B-Chat local inference backend.

Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0  (~2.2 GB, CPU-friendly)
Requires: pip install transformers torch accelerate

Usage
-----
    from shared.tinyllama_llm import TinyLlamaLLM

    llm = TinyLlamaLLM()                      # auto-downloads on first run
    response = llm.generate("Who built Rome?")
    print(response)

CLI (rollouts)
--------------
    python scripts/generate_rollouts.py \\
        --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --max-queries 20 --k 4
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default HuggingFace model id — pin to 1.1B Chat variant
TINYLLAMA_MODEL_ID = os.environ.get(
    "TINYLLAMA_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

# System prompt matching the ReAct rollout format used in generate_rollouts.py
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


class TinyLlamaLLM:
    """
    TinyLlama-1.1B-Chat via HuggingFace transformers pipeline.

    Works on CPU (float32) or GPU (float16 auto-detected).
    Applies the official chat template so the model stays in-distribution.
    """

    def __init__(
        self,
        model_id: str = TINYLLAMA_MODEL_ID,
        device: str = "auto",
        max_new_tokens: int = 256,
        system_prompt: str = _REACT_SYSTEM,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError as exc:
            raise ImportError(
                "TinyLlamaLLM requires 'transformers' and 'torch'.\n"
                "Install with: pip install transformers torch accelerate"
            ) from exc

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt

        _device = "cuda" if (device == "auto" and torch.cuda.is_available()) else "cpu"
        _dtype = torch.float16 if _device == "cuda" else torch.float32
        logger.info("Loading TinyLlama model %s on %s (%s)", model_id, _device, _dtype)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=_dtype,
            device_map="auto" if _device == "cuda" else None,
        )

        self._pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        self._tokenizer = tokenizer
        logger.info("TinyLlama ready.")

    # ------------------------------------------------------------------
    def _build_prompt(self, user_message: str) -> str:
        """
        Apply TinyLlama chat template:
        <|system|>...</s><|user|>...</s><|assistant|>
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_message},
        ]
        try:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback for tokenizers that don't support chat_template
            return (
                f"<|system|>\n{self.system_prompt}</s>\n"
                f"<|user|>\n{user_message}</s>\n"
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
        temperature: Sampling temperature (default 0.7 — balanced creativity).
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
            )
            text: str = outputs[0]["generated_text"].strip()
            latency_ms = (time.perf_counter() - t0) * 1000

            from shared.llm_logging import log_llm_call

            log_llm_call(
                backend="tinyllama",
                model=self.model_id,
                prompt=full_prompt,
                response=text,
                success=True,
                context=context,
                agent=agent,
                latency_ms=latency_ms,
            )
            logger.debug("TinyLlama [%.0f ms]: %s…", latency_ms, text[:80])
            return text

        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            from shared.llm_logging import log_llm_call

            log_llm_call(
                backend="tinyllama",
                model=self.model_id,
                prompt=full_prompt,
                response="",
                success=False,
                error=str(exc),
                context=context,
                agent=agent,
                latency_ms=latency_ms,
            )
            logger.error("TinyLlama generation failed: %s", exc)
            raise


# ---------------------------------------------------------------------------
# Convenience factory used by resolve_llm() in generate_rollouts.py
# ---------------------------------------------------------------------------
def build_tinyllama(
    model_id: str = TINYLLAMA_MODEL_ID,
    max_new_tokens: int = 256,
) -> TinyLlamaLLM:
    """Return a ready-to-use TinyLlamaLLM instance."""
    return TinyLlamaLLM(model_id=model_id, max_new_tokens=max_new_tokens)
