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
AUDIO_DIR="StreamCorrect_assets/StreamCorrect/aishell1_testset"
REFERENCE_FILE="StreamCorrect_assets/StreamCorrect/reference.json"
MODEL_PATH="large-v2.pt"
OUTPUT_DIR="save_dir/streaming_largev3-c05_b4_all_aishell_results_without_ec/"
NUM_FILES=10

# Parallel processing configuration
NUM_WORKERS=4         # Number of parallel workers (set to 1 for sequential processing)
GPUS="0,1,2,3"         # Comma-separated list of GPU IDs to use
# export CUDA_VISIBLE_DEVICES=0,1,2,5,6,7

# # Error corrector configuration (set USE_ERROR_CORRECTOR=true to enable)
USE_ERROR_CORRECTOR="${USE_ERROR_CORRECTOR:-false}"
ERROR_CORRECTOR_CKPT="${ERROR_CORRECTOR_CKPT:-SpeechLMCorrector/ckpts/ultravox_lora_continued_more_erroneous_9/checkpoint-2210}"
ERROR_CORRECTOR_BASE_MODEL="${ERROR_CORRECTOR_BASE_MODEL:-fixie-ai/ultravox-v0_5-llama-3_2-1b}"
ERROR_CORRECTOR_TYPE="${ERROR_CORRECTOR_TYPE:-speechlm}"

# USE_ERROR_CORRECTOR="${USE_ERROR_CORRECTOR:-true}"
# ERROR_CORRECTOR_CKPT="${ERROR_CORRECTOR_CKPT:-SpeechLMCorrector/ckpts/ultravox_8b_lora_finetuned_9/checkpoint-924}"
# ERROR_CORRECTOR_BASE_MODEL="${ERROR_CORRECTOR_BASE_MODEL:-fixie-ai/ultravox-v0_5-llama-3_1-8b}"
# ERROR_CORRECTOR_TYPE="${ERROR_CORRECTOR_TYPE:-speechlm}"

# USE_ERROR_CORRECTOR="${USE_ERROR_CORRECTOR:-true}"
# ERROR_CORRECTOR_CKPT="${ERROR_CORRECTOR_CKPT:-LMCorrector/ckpts/llama_lora_finetuned_4/checkpoint-616}"
# ERROR_CORRECTOR_BASE_MODEL="${ERROR_CORRECTOR_BASE_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
# ERROR_CORRECTOR_TYPE="${ERROR_CORRECTOR_TYPE:-lm}"

echo ""
echo "=========================================="
echo "SimulStreaming Integrated Workflow Example"
echo "=========================================="
echo ""

# Run batch transcription with automatic evaluation
echo ""
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
    --lan \"zh\" \
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
