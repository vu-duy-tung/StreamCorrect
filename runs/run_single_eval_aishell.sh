#!/bin/bash
# Example workflow: Single file transcription with integrated evaluation
# This script demonstrates the complete workflow with automatic evaluation

set -e  # Exit on error

# Configuration
AUDIO_PATH="${AUDIO_PATH:-/data/mino/AISHELL-1/data_aishell/wav/train/S0601/BAC009S0601W0468.wav}"
REFERENCE_FILE="${REFERENCE_FILE:-/data/mino/AISHELL-1/data_aishell/transcript/reference.json}"
MODEL_PATH="${MODEL_PATH:-large-v2.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-./example_output}"
# Make output dir unique per audio (safe for parallel runs)
OUTPUT_DIR="${OUTPUT_DIR%/}/$(basename "${AUDIO_PATH%.*}")"

VAC_CHUNK_SIZE="${VAC_CHUNK_SIZE:-0.5}"
BEAM_SIZE="${BEAM_SIZE:-4}"

export AUDIO_PATH

echo "=========================================="
echo "SimulStreaming Integrated Workflow Example"
echo "=========================================="
echo ""

# Run single file transcription with automatic evaluation
echo "Running single file transcription with automatic evaluation..."
echo "  Audio path: $AUDIO_PATH"
echo "  Reference file: $REFERENCE_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo "  VAC chunk size: $VAC_CHUNK_SIZE seconds"
echo "  Beam size: $BEAM_SIZE"
echo ""

python simulstreaming_whisper.py "$AUDIO_PATH" \
    --model_path "$MODEL_PATH" \
    --logdir "$OUTPUT_DIR" \
    --vac \
    --vac-chunk-size "$VAC_CHUNK_SIZE" \
    --min-chunk-size 0.01 \
    --lan "zh" \
    --beams "$BEAM_SIZE" \
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

