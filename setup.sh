#!/bin/bash
set -e

# 1. Create and activate conda environment
conda create -n StreamCorrect python=3.10 -y
conda activate StreamCorrect

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download model checkpoints and data assets
pip install gdown -q
gdown --folder "https://drive.google.com/drive/folders/11tjxuqyZsUQ7mEULIXFyzdu4UwacBYC0?usp=drive_link"

# 4. Extract model checkpoint
unzip -o StreamCorrect_assets/large-v2-zh.pt.zip
rm -rf __MACOSX

# 5. Extract evaluation dataset
unzip -o StreamCorrect_assets/StreamCorrect.zip -d StreamCorrect_assets/

echo "Setup complete! Activate the environment with: conda activate StreamCorrect"

