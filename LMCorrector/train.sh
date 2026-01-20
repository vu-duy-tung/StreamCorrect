#!/bin/bash

# =============================================================================
# LMCorrector Training Script
# Fine-tune Llama 3.2 1B with LoRA for ASR error correction (text-only)
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

# Paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_FILE="${SCRIPT_DIR}/training_config.yaml"
OUTPUT_DIR="${SCRIPT_DIR}/runs/llama_lora_finetuned"

# Training data (update these paths as needed)
TRAIN_DATA=""  # Path to training JSONL file
EVAL_DATA=""   # Path to evaluation JSONL file (optional)

# Training settings
NUM_GPUS=1                    # Number of GPUs to use
MASTER_PORT=29500             # Port for DDP communication
BATCH_SIZE=4                  # Per-device batch size
GRAD_ACCUM=8                  # Gradient accumulation steps
EPOCHS=3                      # Number of training epochs
LEARNING_RATE="2e-4"          # Learning rate

# Resume training (set to checkpoint path if resuming)
RESUME_CHECKPOINT=""          # e.g., "./runs/llama_lora_finetuned/checkpoint-500"
LOAD_ADAPTER=""               # Path to pre-trained LoRA adapter to continue training from

# =============================================================================
# Environment Setup
# =============================================================================

# Activate conda environment if available
if command -v conda &> /dev/null; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate base 2>/dev/null || true
fi

# Set CUDA device(s) if not already set
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="0"  # Default to GPU 0
fi

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# =============================================================================
# Logging
# =============================================================================

echo "=============================================="
echo "LMCorrector Training Script"
echo "=============================================="
echo "Config file: ${CONFIG_FILE}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Train data: ${TRAIN_DATA:-'Not specified (will use dummy data)'}"
echo "Eval data: ${EVAL_DATA:-'Not specified'}"
echo "Num GPUs: ${NUM_GPUS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Grad accum: ${GRAD_ACCUM}"
echo "Epochs: ${EPOCHS}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Resume checkpoint: ${RESUME_CHECKPOINT:-'None'}"
echo "Load adapter: ${LOAD_ADAPTER:-'None'}"
echo "=============================================="

# =============================================================================
# Create output directory
# =============================================================================

mkdir -p "${OUTPUT_DIR}"

# =============================================================================
# Build training command
# =============================================================================

TRAIN_CMD="python ${SCRIPT_DIR}/training.py"

# Add config file
TRAIN_CMD="${TRAIN_CMD} --config ${CONFIG_FILE}"

# Add training data if specified
if [ -n "${TRAIN_DATA}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --train_data ${TRAIN_DATA}"
fi

# Add eval data if specified
if [ -n "${EVAL_DATA}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --eval_data ${EVAL_DATA}"
fi

# Add output dir
TRAIN_CMD="${TRAIN_CMD} --output_dir ${OUTPUT_DIR}"

# Add resume checkpoint if specified
if [ -n "${RESUME_CHECKPOINT}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --resume_from_checkpoint ${RESUME_CHECKPOINT}"
fi

# Add load adapter if specified
if [ -n "${LOAD_ADAPTER}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --load_adapter ${LOAD_ADAPTER}"
fi

# =============================================================================
# Run training
# =============================================================================

if [ "${NUM_GPUS}" -gt 1 ]; then
    # Multi-GPU training with torchrun
    echo "Starting multi-GPU training with ${NUM_GPUS} GPUs..."
    
    # Find an available port
    while netstat -tuln 2>/dev/null | grep -q ":${MASTER_PORT} " || ss -tuln 2>/dev/null | grep -q ":${MASTER_PORT} "; do
        MASTER_PORT=$((MASTER_PORT + 1))
    done
    
    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=${MASTER_PORT} \
        ${SCRIPT_DIR}/training.py \
        --config ${CONFIG_FILE} \
        ${TRAIN_DATA:+--train_data ${TRAIN_DATA}} \
        ${EVAL_DATA:+--eval_data ${EVAL_DATA}} \
        --output_dir ${OUTPUT_DIR} \
        ${RESUME_CHECKPOINT:+--resume_from_checkpoint ${RESUME_CHECKPOINT}} \
        ${LOAD_ADAPTER:+--load_adapter ${LOAD_ADAPTER}}
else
    # Single-GPU training
    echo "Starting single-GPU training..."
    ${TRAIN_CMD}
fi

# =============================================================================
# Post-training
# =============================================================================

echo "=============================================="
echo "Training complete!"
echo "Model saved to: ${OUTPUT_DIR}"
echo "=============================================="

# Show output directory contents
echo ""
echo "Output directory contents:"
ls -la "${OUTPUT_DIR}"
