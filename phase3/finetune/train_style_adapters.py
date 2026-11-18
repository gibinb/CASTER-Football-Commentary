"""
CASTER Phase 3 -- trains lightweight per-style LoRA adapters (dramatic /
casual / statistical) on top of the already fine-tuned base adapter.

Run qlora_finetune.py FIRST to produce the base adapter, then run this
once per style, using each training pair's own input_json.commentary_style
field (set by data_collection/format_training_pairs.py's infer_style()).

Usage:
    python phase3/finetune/train_style_adapters.py \
        --base-adapter ./caster_base_adapter \
        --style dramatic \
        --data data/commentary/training_pairs.jsonl \
        --output-dir ./caster_adapter_dramatic
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

sys.path.append(str(Path(__file__).resolve().parents[2]))
from phase3.finetune.qlora_finetune import format_example  # reuse identical prompt format


def load_style_subset(path: str, style: str) -> Dataset:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry["input_json"].get("commentary_style") == style:
                rows.append({
                    "event_json": json.dumps(entry["input_json"], default=str),
                    "commentary": entry["output"],
                })
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--base-adapter", required=True,
                         help="path to the general-purpose adapter from qlora_finetune.py")
    parser.add_argument("--style", required=True, choices=["dramatic", "casual", "statistical"])
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    dataset = load_style_subset(args.data, args.style)
    if len(dataset) < 200:
        print(f"[warn] only {len(dataset)} examples for style '{args.style}' -- "
              f"aim for 500+ before trusting this adapter (see Phase 3 deep dive risk notes)")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(base, args.base_adapter, is_trainable=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=25,
        save_steps=200,
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset,
        args=training_args, max_seq_length=256, formatting_func=format_example,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    print(f"style adapter '{args.style}' saved to {args.output_dir} (~28MB expected)")


if __name__ == "__main__":
    main()
