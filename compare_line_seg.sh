#!/bin/bash

TEST_DATA_PATH="../data/medieval-segmentation/src/altos/test"
LINE_CONFIG="line/configs/catmus_ms_line.json"
LAYOUT_CONFIG="layout/configs/catmus_seg_s_16_10.json"
OUTPUT_DIR="runs/results/compare_$(date +%Y%m%d)"

# Create output directory
mkdir -p "$OUTPUT_DIR/layout_pred"

echo "Line segmentation comparison - GT vs layout predictions"
echo "Test data: $TEST_DATA_PATH"
echo "Results in: $OUTPUT_DIR"

# 1. Generate layout predictions
echo "1. Generating layout predictions..."
python run.py layout --function predict --configs "$LAYOUT_CONFIG" --corpus_path "$TEST_DATA_PATH" --output "$OUTPUT_DIR/layout_pred"

# 2. Evaluate line segmentation with GT zones
echo "2. Evaluating lines with ground truth zones..."
python run.py line --function eval --configs "$LINE_CONFIG" --data_path "$TEST_DATA_PATH" --output "$OUTPUT_DIR/metrics_gt.csv"

# 3. Evaluate line segmentation with predicted zones
echo "3. Evaluating lines with predicted zones..."
python run.py line --function eval --configs "$LINE_CONFIG" --data_path "$OUTPUT_DIR/layout_pred" --output "$OUTPUT_DIR/metrics_pred.csv"

# 4. Clean up - remove copied images to save space
echo "4. Cleaning up temporary image files..."
find "$OUTPUT_DIR/layout_pred" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) -delete

# 5. Display results
echo -e "\n=== RESULTS ==="
echo "GT metrics:"
cat "$OUTPUT_DIR/metrics_gt.csv"
echo -e "\nPredicted layout metrics:"
cat "$OUTPUT_DIR/metrics_pred.csv"