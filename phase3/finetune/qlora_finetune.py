"""
CASTER Phase 3 -- QLoRA fine-tuning of Llama 3.2 3B on training_pairs.jsonl
(produced by data_collection/format_training_pairs.py).

Designed to run on a free Google Colab T4 GPU.

Colab setup:
    !pip install transformers peft bitsandbytes datasets accelerate trl -q

Usage:
    python phase3/finetune/qlora_finetune.py \
        --data data/commentary/training_pairs.jsonl \
        --output-dir ./caster_base_adapter \
        --epochs 3

Checkpoints save every --save-steps steps, so an interrupted Colab
session loses at most one checkpoint's worth of progress, not the
whole run (see CASTER_Phase3_DeepDive.docx Section 4.2).
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

sys.path.append(str(Path(__file__).resolve().parents[2]))
from phase3.inference.prompt_builder import SYSTEM_PROMPT  # single source of truth for the prompt


def load_dataset(path: str) -> Dataset:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            rows.append({
                "event_json": json.dumps(entry["input_json"], default=str),
                "commentary": entry["output"],
            })
    return Dataset.from_list(rows)


def format_example(example: dict) -> str:
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\nEvent: {example['event_json']}\n"
        f"<|assistant|>\n{example['commentary']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--base-model", default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save-steps", type=int, default=500)
    args = parser.parse_args()

    dataset = load_dataset(args.data)
    split = dataset.train_test_split(test_size=0.05, seed=42)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        logging_steps=50,
        save_steps=args.save_steps,
        save_total_limit=3,
        evaluation_strategy="steps",
        eval_steps=args.save_steps,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        args=training_args,
        max_seq_length=256,
        formatting_func=format_example,
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
