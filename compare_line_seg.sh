#!/bin/bash

TEST_DATA_PATH="../data/medieval-segmentation/src/altos/test"
LINE_CONFIG="line/configs/catmus_ms_line.json"
LAYOUT_CONFIG="layout/configs/catmus_seg_s_16_10.json"
OUTPUT_DIR="runs/results/compare_$(date +%Y%m%d)"

# Create output directory
mkdir -p "$OUTPUT_DIR/layout_pred"
mkdir -p "$OUTPUT_DIR/line_pred"

echo "Line segmentation comparison - GT vs layout predictions"
echo "Test data: $TEST_DATA_PATH"
echo "Results in: $OUTPUT_DIR"

# Evaluate line segmentation with GT zones
echo "1. Evaluating lines with ground truth zones..."
python run.py line --function eval --configs "$LINE_CONFIG" --data_path "$TEST_DATA_PATH" --output "$OUTPUT_DIR/metrics_gt.csv"

# Generate layout predictions
echo "2. Generating layout predictions..."
python run.py layout --function predict --configs "$LAYOUT_CONFIG" --pred_path "$TEST_DATA_PATH" --output "$OUTPUT_DIR/layout_pred"

# Predict line segmentation with predicted zones
echo "3. Predicting lines with predicted zones..."
python run.py line --function predict --configs "$LINE_CONFIG" --pred_path "$OUTPUT_DIR/layout_pred" --output "$OUTPUT_DIR/line_pred"

# Evaluate line segmentation with predicted zones
echo "4. Scoring lines with predicted zones..."
python run.py line --function score --pred_path "$OUTPUT_DIR/line_pred" --data_path "$TEST_DATA_PATH" --output "$OUTPUT_DIR/metrics_pred.csv"

# Display results
echo -e "\n=== RESULTS ==="
echo "GT metrics:"
cat "$OUTPUT_DIR/metrics_gt.csv"
echo -e "\nPredicted layout metrics:"
cat "$OUTPUT_DIR/metrics_pred.csv"