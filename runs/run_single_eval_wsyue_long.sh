#!/bin/bash
# Example workflow: Single file transcription with integrated evaluation
# This script demonstrates the complete workflow with automatic evaluation

set -e  # Exit on error

# Configuration
AUDIO_PATH="/home/duy/PlayWithMino/SimulStreaming/save_dir/data/WSYue-ASR-eval/Long/wav/0000093541.wav"
REFERENCE_FILE="/home/duy/PlayWithMino/SimulStreaming/save_dir/data/WSYue-ASR-eval/Long/reference.json"
MODEL_PATH="medium.pt"
OUTPUT_DIR="./example_output"

echo "=========================================="
echo "SimulStreaming Integrated Workflow Example"
echo "=========================================="
echo ""

# Run single file transcription with automatic evaluation
echo "Running single file transcription with automatic evaluation..."
echo "  Audio path: $AUDIO_PATH"
echo "  Reference file: $REFERENCE_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo ""

python simulstreaming_whisper.py "$AUDIO_PATH" \
    --model_path "$MODEL_PATH" \
    --logdir "$OUTPUT_DIR" \
    --vac \
    --vac-chunk-size 0.5 \
    --min-chunk-size 0.01 \
    --lan "yue" \
    --beams 4 \
    --frame_threshold 20 \
    --reference-file "$REFERENCE_FILE" \
    --log-level DEBUG \
    --comp_unaware

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
print(f"  Full JSON: {path}")
PY
fi

