#!/bin/bash
# Single file transcription using Qwen3-ASR-1.7B as ASR backbone
# with optional error correction (Mandarin / AISHELL)

set -e

CONDA_BASE="$(conda info --base 2>/dev/null || echo /data/mino/anaconda3)"
PYTHON="${CONDA_BASE}/envs/StreamCorrect/bin/python"
[ ! -x "$PYTHON" ] && echo "ERROR: Python not found at $PYTHON" && exit 1

# Configuration
AUDIO_PATH="${AUDIO_PATH:-StreamCorrect_assets/StreamCorrect/aishell1_testset_snr0/BAC009S0764W0216.wav}"
REFERENCE_FILE="${REFERENCE_FILE:-StreamCorrect_assets/StreamCorrect/reference.json}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-ASR-1.7B}"
OUTPUT_DIR="${OUTPUT_DIR:-./example_output}"
OUTPUT_DIR="${OUTPUT_DIR%/}/$(basename "${AUDIO_PATH%.*}")_qwen3asr"

VAC_CHUNK_SIZE="${VAC_CHUNK_SIZE:-0.5}"
BEAM_SIZE="${BEAM_SIZE:-4}"

# Error corrector configuration
USE_ERROR_CORRECTOR="${USE_ERROR_CORRECTOR:-true}"
ERROR_CORRECTOR_CKPT="${ERROR_CORRECTOR_CKPT:-StreamCorrect_assets/StreamCorrect/error_corrector_ckpt}"
ERROR_CORRECTOR_BASE_MODEL="${ERROR_CORRECTOR_BASE_MODEL:-fixie-ai/ultravox-v0_5-llama-3_2-1b}"
ERROR_CORRECTOR_TYPE="${ERROR_CORRECTOR_TYPE:-speechlm}"

export AUDIO_PATH

echo "==========================================="
echo "Qwen3-ASR Streaming + Error Corrector (AISHELL / Mandarin)"
echo "==========================================="
echo "  Audio: $AUDIO_PATH"
echo "  Reference: $REFERENCE_FILE"
echo "  Model: $MODEL_PATH"
echo "  Beams: $BEAM_SIZE"
echo "  VAC chunk size: $VAC_CHUNK_SIZE"
echo "  Error corrector: $USE_ERROR_CORRECTOR"
if [ "$USE_ERROR_CORRECTOR" = "true" ] && [ -n "$ERROR_CORRECTOR_CKPT" ]; then
    echo "  Error corrector ckpt: $ERROR_CORRECTOR_CKPT"
fi
echo ""

CMD="$PYTHON qwen3asr_streaming.py \"$AUDIO_PATH\" \
    --model_path \"$MODEL_PATH\" \
    --vac \
    --logdir \"$OUTPUT_DIR\" \
    --vac-chunk-size $VAC_CHUNK_SIZE \
    --min-chunk-size 0.01 \
    --lan \"zh\" \
    --beams $BEAM_SIZE \
    --reference-file \"$REFERENCE_FILE\" \
    --log-level DEBUG \
    --comp_unaware"

if [ "$USE_ERROR_CORRECTOR" = "true" ]; then
    CMD="$CMD --use-error-corrector"
    CMD="$CMD --error-corrector-type \"$ERROR_CORRECTOR_TYPE\""
    [ -n "$ERROR_CORRECTOR_CKPT" ] && CMD="$CMD --error-corrector-ckpt \"$ERROR_CORRECTOR_CKPT\""
    [ -n "$ERROR_CORRECTOR_BASE_MODEL" ] && CMD="$CMD --error-corrector-base-model \"$ERROR_CORRECTOR_BASE_MODEL\""
fi

eval $CMD

echo ""
echo "==========================================="
echo "Workflow Complete!"
echo "==========================================="

if [ -f "$OUTPUT_DIR/evaluation_result.json" ]; then
    echo "Evaluation summary:"
    OUTPUT_FILE="$OUTPUT_DIR/evaluation_result.json" $PYTHON - <<'PY'
import json, os
path = os.environ["OUTPUT_FILE"]
with open(path, encoding="utf-8") as f:
        data = json.load(f)
print(f"  CER: {data.get('cer', 'N/A')}")
print(f"  MER: {data.get('mer', 'N/A')}")
print(f"  FTL: {data.get('first_token_latency_ms', 'N/A')} ms")
print(f"  LTL: {data.get('last_token_latency_ms', 'N/A')} ms")
print(f"  Full JSON: {path}")
PY
fi
