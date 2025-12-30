#!/bin/bash
# Train error corrector model using DDP with 6 GPUs
# 
# Usage:
#   cd /home/duy/PlayWithMino/SimulStreaming
#   bash train_ddp.sh

set -e

# Configuration
NUM_GPUS=8
CONFIG_FILE="error_corrector/training/configs/training_configs.yaml"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TOKENIZERS_PARALLELISM=false

# Optional: Set CUDA visible devices if needed (uncomment and modify as needed)
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

echo "=============================================="
echo "Training Error Corrector with DDP"
echo "=============================================="
echo "Number of GPUs: ${NUM_GPUS}"
echo "Config file: ${CONFIG_FILE}"
echo "=============================================="

# Run training with torchrun (PyTorch's distributed launcher)
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    -m error_corrector.training.train \
    --config_path ${CONFIG_FILE}
