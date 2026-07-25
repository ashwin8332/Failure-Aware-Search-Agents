"""Load live inference settings from configs/inference.yaml + environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PATH = _ROOT / "configs" / "inference.yaml"


def load_inference_config() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "backend": "mock",
        "ollama_model": os.environ.get("OLLAMA_MODEL", "phi3:mini"),
        "ollama_host": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        "use_dpo": False,
        "dpo_checkpoint": os.environ.get("DPO_CHECKPOINT", "./checkpoints/dpo-failure-aware"),
        "agents": {
            "search_model": None,
            "verification_model": None,
            "composer_model": None,
        },
    }

    if _DEFAULT_PATH.exists():
        try:
            import yaml

            with open(_DEFAULT_PATH, encoding="utf-8") as f:
                file_cfg = yaml.safe_load(f) or {}
            cfg.update({k: v for k, v in file_cfg.items() if v is not None})
            if file_cfg.get("agents"):
                cfg["agents"].update(file_cfg["agents"])
        except Exception:
            pass

    if os.environ.get("INFERENCE_BACKEND"):
        cfg["backend"] = os.environ["INFERENCE_BACKEND"].strip().lower()
    if os.environ.get("USE_DPO", "").lower() in ("1", "true", "yes"):
        cfg["use_dpo"] = True
    if os.environ.get("DPO_CHECKPOINT"):
        cfg["dpo_checkpoint"] = os.environ["DPO_CHECKPOINT"]

    return cfg
