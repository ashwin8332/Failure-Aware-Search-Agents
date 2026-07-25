"""
DPO fine-tuning with QLoRA using HuggingFace TRL DPOTrainer.

Usage:
  python training/dpo_train.py --data_dir data/preference_pairs --epochs 1
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOConfig, DPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_and_tokenizer(base_model: str, lora_cfg: dict):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    if torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora_cfg.get("lora_r", 64),
        lora_alpha=lora_cfg.get("lora_alpha", 128),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO fine-tuning")
    parser.add_argument("--config", default="configs/dpo_config.yaml")
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--wandb_project", default="failure-aware-dpo")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples for smoke test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_model = args.base_model or os.environ.get("BASE_MODEL", cfg["base_model"])
    data_dir = args.data_dir or cfg["data_dir"]
    output_dir = args.output_dir or cfg["output_dir"]
    beta = args.beta if args.beta is not None else cfg["beta"]
    epochs = args.epochs if args.epochs is not None else cfg["num_train_epochs"]
    seed = cfg.get("seed", 42)
    set_seed(seed)

    train_path = Path(data_dir) / "train_pairs.jsonl"
    val_path = Path(data_dir) / "val_pairs.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing {train_path}. Run scripts/build_preference_pairs.py first."
        )

    dataset = load_dataset("json", data_files={"train": str(train_path), "val": str(val_path)})
    if args.max_samples:
        dataset["train"] = dataset["train"].select(range(min(args.max_samples, len(dataset["train"]))))

    model, tokenizer = build_model_and_tokenizer(base_model, cfg)

    # Reference model: frozen copy for DPO KL term
    ref_model = None
    if torch.cuda.is_available():
        ref_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            device_map="auto",
            trust_remote_code=True,
        )

    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        import wandb

        wandb.init(project=args.wandb_project, config={**cfg, "base_model": base_model})

    training_args = DPOConfig(
        output_dir=output_dir,
        beta=beta,
        learning_rate=cfg["learning_rate"],
        num_train_epochs=epochs,
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        max_length=cfg["max_length"],
        max_prompt_length=cfg["max_prompt_length"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        bf16=torch.cuda.is_available(),
        fp16=False,
        optim="paged_adamw_32bit" if torch.cuda.is_available() else "adamw_torch",
        report_to="wandb" if use_wandb else "none",
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"] if "val" in dataset else None,
        processing_class=tokenizer,
    )

    logger.info("Starting DPO training on %d pairs", len(dataset["train"]))
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    if hasattr(trainer.model, "merge_and_unload"):
        try:
            merged = trainer.model.merge_and_unload()
            merged.save_pretrained(f"{output_dir}/merged")
            logger.info("Saved merged model to %s/merged", output_dir)
        except Exception as exc:
            logger.warning("Could not merge LoRA weights: %s", exc)

    logger.info("Training complete. Checkpoints saved to %s", output_dir)


if __name__ == "__main__":
    main()
