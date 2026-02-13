#!/bin/bash
# Ultravox LoRA Fine-tuning Script
# Usage:
#   ./train.sh                          # Single GPU
#   ./train.sh --gpus 4                 # Multi-GPU DDP
#   ./train.sh --adapter ./checkpoint   # Resume from adapter

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="training_config.yaml"
NUM_GPUS=1
ADAPTER=""
OUTPUT_DIR=""
PORT=29500

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)     NUM_GPUS="$2"; shift 2 ;;
        --config)   CONFIG="$2"; shift 2 ;;
        --adapter)  ADAPTER="$2"; shift 2 ;;
        --output)   OUTPUT_DIR="$2"; shift 2 ;;
        --port)     PORT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--gpus N] [--config FILE] [--adapter PATH] [--output DIR]"
            exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Build arguments
ARGS="--config ${SCRIPT_DIR}/${CONFIG}"
[ -n "$ADAPTER" ] && ARGS="$ARGS --load_adapter $ADAPTER"
[ -n "$OUTPUT_DIR" ] && ARGS="$ARGS --output_dir $OUTPUT_DIR"

cd "$SCRIPT_DIR"

echo "=== Ultravox LoRA Training ==="
echo "GPUs: $NUM_GPUS | Config: $CONFIG"

if [ "$NUM_GPUS" -gt 1 ]; then
    # Find an available port
    while netstat -tuln 2>/dev/null | grep -q ":${PORT} " || ss -tuln 2>/dev/null | grep -q ":${PORT} "; do
        PORT=$((PORT + 1))
    done
    echo "Using port: $PORT"
    
    # NCCL workaround for H20 GPUs: disable P2P and SHM to avoid CUDA IPC issues
    # This forces NCCL to use socket-based communication instead
    export NCCL_P2P_DISABLE=1
    export NCCL_SHM_DISABLE=1
    
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$PORT training.py $ARGS
else
    python training.py $ARGS
fi

echo "=== Training Complete ==="
