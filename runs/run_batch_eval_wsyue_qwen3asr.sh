#!/bin/bash
# Batch transcription using Qwen3-ASR-1.7B as ASR backbone
# with optional error correction (Cantonese / WSYue)
# Supports parallel processing with multiple GPUs

set -e  # Exit on error

# Resolve Python from StreamCorrect conda environment
CONDA_BASE="$(conda info --base 2>/dev/null || echo /data/mino/anaconda3)"
PYTHON="${CONDA_BASE}/envs/StreamCorrect/bin/python"
[ ! -x "$PYTHON" ] && echo "ERROR: Python not found at $PYTHON" && exit 1

# Configuration
AUDIO_DIR="${AUDIO_DIR:-/mnt/nas_disk1/duy1/PlayWithMino/SimulStreaming/save_dir/data/WSYue-ASR-eval/Short/wav_}"
REFERENCE_FILE="${REFERENCE_FILE:-/mnt/nas_disk1/duy1/PlayWithMino/SimulStreaming/save_dir/data/WSYue-ASR-eval/Short/content.json}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-ASR-1.7B}"
OUTPUT_DIR="${OUTPUT_DIR:-save_dir/qwen3asr_wsyue_results_100}"
NUM_FILES="${NUM_FILES:-100}"

VAC_CHUNK_SIZE="${VAC_CHUNK_SIZE:-0.5}"
BEAM_SIZE="${BEAM_SIZE:-4}"

# Parallel processing configuration
NUM_WORKERS="${NUM_WORKERS:-5}"          # Number of parallel workers
GPUS="${GPUS:-3,4,5,6,7}"              # Comma-separated GPU IDs

# Error corrector configuration
USE_ERROR_CORRECTOR="${USE_ERROR_CORRECTOR:-false}"
ERROR_CORRECTOR_CKPT="${ERROR_CORRECTOR_CKPT:-/data/mino/model_ckpts/wsyue_qwen2audio_7b_lora_finetuned_mix_4/checkpoint-3492}"
ERROR_CORRECTOR_BASE_MODEL="${ERROR_CORRECTOR_BASE_MODEL:-Qwen/Qwen2-Audio-7B-Instruct}"
ERROR_CORRECTOR_TYPE="${ERROR_CORRECTOR_TYPE:-speechlm}"

echo "==========================================="
echo "Qwen3-ASR Batch Streaming + Error Corrector (WSYue / Cantonese)"
echo "==========================================="
echo ""
echo "  Audio directory: $AUDIO_DIR"
echo "  Number of files: $NUM_FILES"
echo "  Reference file:  $REFERENCE_FILE"
echo "  Model:           $MODEL_PATH"
echo "  Output dir:      $OUTPUT_DIR"
echo "  Beams:           $BEAM_SIZE"
echo "  VAC chunk size:  $VAC_CHUNK_SIZE"
echo "  Parallel workers:$NUM_WORKERS"
echo "  GPUs:            $GPUS"
echo "  Error corrector: $USE_ERROR_CORRECTOR"
if [ "$USE_ERROR_CORRECTOR" = "true" ] && [ -n "$ERROR_CORRECTOR_CKPT" ]; then
    echo "  Corrector ckpt:  $ERROR_CORRECTOR_CKPT"
fi
echo ""

# Build command
CMD="$PYTHON qwen3asr_streaming.py \"$AUDIO_DIR\" \
    --model_path \"$MODEL_PATH\" \
    --logdir \"$OUTPUT_DIR\" \
    --vac \
    --vac-chunk-size $VAC_CHUNK_SIZE \
    --min-chunk-size 0.01 \
    --lan \"yue\" \
    --beams $BEAM_SIZE \
    --reference-file \"$REFERENCE_FILE\" \
    --log-level INFO \
    --num-audios $NUM_FILES \
    --num-workers $NUM_WORKERS \
    --gpus \"$GPUS\" \
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
echo "Batch Workflow Complete!"
echo "==========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "  - batch_transcriptions.json (all transcriptions)"
echo "  - evaluation_results.json   (CER/MER/latency metrics)"
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
