#!/bin/bash
set -e

ENV_NAME="StreamCorrect"

# 1. Create conda environment
conda create -n "$ENV_NAME" python=3.10 -y

# Initialize conda for this shell session
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download model checkpoints and data assets
git clone https://huggingface.co/datasets/playwithmino/StreamCorrect
mv StreamCorrect StreamCorrect_assets

# 4. Extract model checkpoint
unzip -o StreamCorrect_assets/large-v2-zh.pt.zip
rm -rf __MACOSX

# 5. Extract evaluation dataset
unzip -o StreamCorrect_assets/StreamCorrect.zip -d StreamCorrect_assets/

echo "Setup complete! Activate the environment with: conda activate $ENV_NAME"

