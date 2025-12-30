#!/bin/bash
# Continue training error corrector model from checkpoint
# 
# Usage:
#   cd /home/duy/PlayWithMino/SimulStreaming
#   bash continue_train.sh

set -e

# Configuration
NUM_GPUS=8
CONFIG_FILE="error_corrector/training/configs/training_configs.yaml"
CHECKPOINT_PATH="runs/exp--2025-12-29--19-04-25/checkpoint-688"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TOKENIZERS_PARALLELISM=false

echo "=============================================="
echo "Continue Training Error Corrector from Checkpoint"
echo "=============================================="
echo "Number of GPUs: ${NUM_GPUS}"
echo "Config file: ${CONFIG_FILE}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Additional epochs: 7"
echo "=============================================="

# Run training with torchrun (PyTorch's distributed launcher)
# --model_load_dir: Load model weights from checkpoint
# --num_epochs: Train for 4 additional epochs
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    -m error_corrector.training.train \
    --config_path ${CONFIG_FILE} \
    --model_load_dir ${CHECKPOINT_PATH} \
    --num_epochs 8
