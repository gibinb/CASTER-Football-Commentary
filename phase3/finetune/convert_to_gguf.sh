#!/usr/bin/env bash
# CASTER Phase 3 -- converts a merged HuggingFace model directory (see
# merge_adapter.py) into a 4-bit quantized GGUF file that llama.cpp can
# run on CPU. Output lands in phase3/models/, exactly where
# inference/local_llm.py looks for it.
#
# Usage:
#   ./convert_to_gguf.sh ./caster_dramatic_merged caster_dramatic_q4.gguf
#
# Prerequisite (one-time setup):
#   git clone https://github.com/ggerganov/llama.cpp
#   cd llama.cpp && make
#   pip install -r llama.cpp/requirements.txt

set -euo pipefail

MERGED_MODEL_DIR="${1:?usage: convert_to_gguf.sh <merged_model_dir> <output.gguf>}"
OUTPUT_GGUF="${2:?usage: convert_to_gguf.sh <merged_model_dir> <output.gguf>}"
LLAMACPP_DIR="${LLAMACPP_DIR:-./llama.cpp}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/../models"

mkdir -p "$OUTPUT_DIR"

echo "[1/2] converting HF model -> f16 GGUF ..."
python "$LLAMACPP_DIR/convert_hf_to_gguf.py" "$MERGED_MODEL_DIR" \
    --outfile "$OUTPUT_DIR/tmp_f16.gguf" \
    --outtype f16

echo "[2/2] quantizing to 4-bit (q4_K_M) ..."
"$LLAMACPP_DIR/llama-quantize" \
    "$OUTPUT_DIR/tmp_f16.gguf" \
    "$OUTPUT_DIR/$OUTPUT_GGUF" \
    q4_K_M

rm -f "$OUTPUT_DIR/tmp_f16.gguf"
echo "done -> $OUTPUT_DIR/$OUTPUT_GGUF"
