#!/bin/bash
# Example workflow: Batch transcription with integrated evaluation
# This script demonstrates the complete workflow with automatic evaluation
# Supports parallel processing with multiple GPUs and error correction

set -e  # Exit on error

# Resolve Python from StreamCorrect conda environment
CONDA_BASE="$(conda info --base 2>/dev/null || echo /data/mino/anaconda3)"
PYTHON="${CONDA_BASE}/envs/StreamCorrect/bin/python"
[ ! -x "$PYTHON" ] && echo "ERROR: Python not found at $PYTHON" && exit 1

# Configuration
AUDIO_DIR="/home/duy/PlayWithMino/SimulStreaming/save_dir/data/WSYue-ASR-eval/Short/wav_"
REFERENCE_FILE="/home/duy/PlayWithMino/SimulStreaming/save_dir/data/WSYue-ASR-eval/Short/content.json"
MODEL_PATH="medium.pt"
OUTPUT_DIR="save_dir/streaming_medium-yue-50_05_with_ec/"
NUM_FILES=50

# Parallel processing configuration
NUM_WORKERS=3          # Number of parallel workers (set to 1 for sequential processing)
GPUS="5,6,7"         # Comma-separated list of GPU IDs to use

# Error corrector configuration (set USE_ERROR_CORRECTOR=true to enable)
USE_ERROR_CORRECTOR="${USE_ERROR_CORRECTOR:-true}"
ERROR_CORRECTOR_CKPT="${ERROR_CORRECTOR_CKPT:-SpeechLMCorrector/ckpts/wsyue_ultravox_1b_lora_finetuned_6/checkpoint-2830}"
ERROR_CORRECTOR_BASE_MODEL="${ERROR_CORRECTOR_BASE_MODEL:-fixie-ai/ultravox-v0_5-llama-3_2-1b}"
# Error corrector type: "speechlm" (audio+text Ultravox) or "lm" (text-only Llama)
ERROR_CORRECTOR_TYPE="${ERROR_CORRECTOR_TYPE:-speechlm}"

echo "=========================================="
echo "SimulStreaming Integrated Workflow Example"
echo "=========================================="
echo ""

# Run batch transcription with automatic evaluation
echo "Running batch transcription with automatic evaluation..."
echo "  Audio directory: $AUDIO_DIR"
echo "  Number of files: $NUM_FILES"
echo "  Reference file: $REFERENCE_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo "  Parallel workers: $NUM_WORKERS"
echo "  GPUs: $GPUS"
echo "  Use error corrector: $USE_ERROR_CORRECTOR"
echo ""

# Build command with optional error corrector flags
CMD="$PYTHON simulstreaming_whisper.py \"$AUDIO_DIR\" \
    --model_path \"$MODEL_PATH\" \
    --logdir \"$OUTPUT_DIR\" \
    --vac \
    --vac-chunk-size 0.5 \
    --min-chunk-size 0.01 \
    --reference-file \"$REFERENCE_FILE\" \
    --log-level INFO \
    --lan \"yue\" \
    --beams 4 \
    --frame_threshold 20 \
    --num-audios $NUM_FILES \
    --num-workers $NUM_WORKERS \
    --gpus \"$GPUS\" \
    --comp_unaware"

# Add error corrector flags if enabled
if [ "$USE_ERROR_CORRECTOR" = "true" ]; then
    CMD="$CMD --use-error-corrector"
    CMD="$CMD --error-corrector-type \"$ERROR_CORRECTOR_TYPE\""
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
echo "Results saved to: $OUTPUT_DIR"
echo "  - batch_transcriptions.json (all transcriptions)"
echo "  - evaluation_results.json (CER metrics)"
echo ""
echo "Quick results:"
echo "  Average CER: $(cat $OUTPUT_DIR/evaluation_results.json | grep -o '"average_cer": [0-9.]*' | cut -d' ' -f2)"
echo "  Average MER: $(cat $OUTPUT_DIR/evaluation_results.json | grep -o '"average_mer": [0-9.]*' | cut -d' ' -f2)"
echo "  Average FTL: $(cat $OUTPUT_DIR/evaluation_results.json | grep -o '"average_first_token_latency_ms": [0-9.]*' | cut -d' ' -f2) ms"
echo "  Average LTL: $(cat $OUTPUT_DIR/evaluation_results.json | grep -o '"average_last_token_latency_ms": [0-9.]*' | cut -d' ' -f2) ms"
echo ""
echo "View detailed results:"
echo "  cat $OUTPUT_DIR/evaluation_results.json | jq '.per_file_results[] | {file: .file, cer: .cer, mer: .mer}'"
echo ""

