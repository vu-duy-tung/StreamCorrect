#!/bin/bash
# Example workflow: Single file transcription with integrated evaluation
# This script demonstrates the complete workflow with automatic evaluation
# Supports error correction

set -e  # Exit on error

# Configuration
AUDIO_PATH="${AUDIO_PATH:-/home/duy/PlayWithMino/SimulStreaming/save_dir/data/WSYue-ASR-eval/Short/wav_/0000004453.wav}"
REFERENCE_FILE="${REFERENCE_FILE:-/home/duy/PlayWithMino/SimulStreaming/save_dir/data/WSYue-ASR-eval/Short/content.json}"
MODEL_PATH="${MODEL_PATH:-medium.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-./example_output}"
# Make output dir unique per audio (safe for parallel runs)
OUTPUT_DIR="${OUTPUT_DIR%/}/$(basename "${AUDIO_PATH%.*}")"

VAC_CHUNK_SIZE="${VAC_CHUNK_SIZE:-0.5}"
BEAM_SIZE="${BEAM_SIZE:-4}"

# Error corrector configuration
USE_ERROR_CORRECTOR="${USE_ERROR_CORRECTOR:-false}"
ERROR_CORRECTOR_CKPT="${ERROR_CORRECTOR_CKPT:-SpeechLMCorrector/ultravox_lora_continued_more_erroneous_6/checkpoint-1158}"
ERROR_CORRECTOR_BASE_MODEL="${ERROR_CORRECTOR_BASE_MODEL:-fixie-ai/ultravox-v0_5-llama-3_2-1b}"

export AUDIO_PATH

echo "=========================================="
echo "SimulStreaming Integrated Workflow Example"
echo "=========================================="
echo ""

# Run single file transcription with automatic evaluation
echo "Running single file transcription with automatic evaluation..."
echo "  Audio path: $AUDIO_PATH"
echo "  Reference file: $REFERENCE_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo "  VAC chunk size: $VAC_CHUNK_SIZE seconds"
echo "  Beam size: $BEAM_SIZE"
echo "  Use error corrector: $USE_ERROR_CORRECTOR"
if [ "$USE_ERROR_CORRECTOR" = "true" ] && [ -n "$ERROR_CORRECTOR_CKPT" ]; then
    echo "  Error corrector checkpoint: $ERROR_CORRECTOR_CKPT"
fi
echo ""

# Build command with optional error corrector flags
CMD="python simulstreaming_whisper.py \"$AUDIO_PATH\" \
    --model_path \"$MODEL_PATH\" \
    --logdir \"$OUTPUT_DIR\" \
    --vac \
    --vac-chunk-size $VAC_CHUNK_SIZE \
    --min-chunk-size 0.01 \
    --lan \"yue\" \
    --beams $BEAM_SIZE \
    --frame_threshold 20 \
    --reference-file \"$REFERENCE_FILE\" \
    --log-level DEBUG \
    --comp_unaware"

# Add error corrector flags if enabled
if [ "$USE_ERROR_CORRECTOR" = "true" ]; then
    CMD="$CMD --use-error-corrector"
    if [ -n "$ERROR_CORRECTOR_CKPT" ]; then
        CMD="$CMD --error-corrector-ckpt \"$ERROR_CORRECTOR_CKPT\""
    fi
    if [ -n "$ERROR_CORRECTOR_BASE_MODEL" ]; then
        CMD="$CMD --error-corrector-base-model \"$ERROR_CORRECTOR_BASE_MODEL\""
    fi
fi

eval $CMD

echo ""
echo "=========================================="
echo "Workflow Complete!"
echo "=========================================="
echo ""

if [ -f "$OUTPUT_DIR/evaluation_result.json" ]; then
    echo "Evaluation summary:"
    OUTPUT_FILE="$OUTPUT_DIR/evaluation_result.json" python - <<'PY'
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

