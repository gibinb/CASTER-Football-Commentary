"""
CASTER Phase 3 -- merges a trained LoRA adapter back into the base Llama
3.2 3B weights, producing a standalone model directory ready for GGUF
conversion. llama.cpp needs full merged weights, not base + adapter
separately, so this is required before convert_to_gguf.sh.

Usage:
    python phase3/finetune/merge_adapter.py \
        --base meta-llama/Llama-3.2-3B \
        --adapter ./caster_adapter_dramatic \
        --output ./caster_dramatic_merged
"""
import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"loading base model {args.base} ...")
    base_model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.base)

    print(f"applying adapter {args.adapter} ...")
    model = PeftModel.from_pretrained(base_model, args.adapter)

    print("merging adapter weights into base model ...")
    merged = model.merge_and_unload()

    merged.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"merged model saved to {args.output} -- convert with convert_to_gguf.sh next")


if __name__ == "__main__":
    main()
