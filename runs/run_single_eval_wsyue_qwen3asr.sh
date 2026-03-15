#!/bin/bash
# Single file transcription using Qwen3-ASR-1.7B as ASR backbone
# with optional error correction via Qwen2-Audio

set -e

CONDA_BASE="$(conda info --base 2>/dev/null || echo /data/mino/anaconda3)"
PYTHON="${CONDA_BASE}/envs/StreamCorrect/bin/python"
[ ! -x "$PYTHON" ] && echo "ERROR: Python not found at $PYTHON" && exit 1

# Configuration
AUDIO_PATH="${AUDIO_PATH:-/mnt/nas_disk1/duy1/PlayWithMino/SimulStreaming/save_dir/data/WSYue-ASR-eval/Short/wav_/xq0045690_251230_260310.wav}"
REFERENCE_FILE="${REFERENCE_FILE:-/mnt/nas_disk1/duy1/PlayWithMino/SimulStreaming/save_dir/data/WSYue-ASR-eval/Short/content.json}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-ASR-1.7B}"
OUTPUT_DIR="${OUTPUT_DIR:-./example_output}"
OUTPUT_DIR="${OUTPUT_DIR%/}/$(basename "${AUDIO_PATH%.*}")_qwen3asr"

VAC_CHUNK_SIZE="${VAC_CHUNK_SIZE:-0.5}"
BEAM_SIZE="${BEAM_SIZE:-4}"

# Error corrector configuration
USE_ERROR_CORRECTOR="${USE_ERROR_CORRECTOR:-true}"
ERROR_CORRECTOR_CKPT="${ERROR_CORRECTOR_CKPT:-/data/mino/model_ckpts/wsyue_qwen2audio_7b_lora_finetuned_mix_4/checkpoint-3492}"
ERROR_CORRECTOR_BASE_MODEL="${ERROR_CORRECTOR_BASE_MODEL:-Qwen/Qwen2-Audio-7B-Instruct}"
ERROR_CORRECTOR_TYPE="${ERROR_CORRECTOR_TYPE:-speechlm}"

export AUDIO_PATH

echo "==========================================="
echo "Qwen3-ASR Streaming + Error Corrector"
echo "==========================================="
echo "  Audio: $AUDIO_PATH"
echo "  Model: $MODEL_PATH"
echo "  Beams: $BEAM_SIZE"
echo "  Error corrector: $USE_ERROR_CORRECTOR"
echo ""

CMD="$PYTHON qwen3asr_streaming.py \"$AUDIO_PATH\" \
    --model_path \"$MODEL_PATH\" \
    --logdir \"$OUTPUT_DIR\" \
    --vac \
    --vac-chunk-size $VAC_CHUNK_SIZE \
    --min-chunk-size 0.01 \
    --lan \"yue\" \
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
PY
fi
