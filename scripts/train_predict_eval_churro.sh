#!/bin/bash

# Script: train_predict_eval_churro.sh
# Usage: ./train_predict_eval_churro.sh [experiment_name] [epochs] [batch_size] [grad_accum] [lora_r] [learning_rate]

set -e  # Exit on error

# ========================================
# CONFIGURATION PARAMETERS
# ========================================

# Default values (can be overridden by command-line arguments)
EXPERIMENT_NAME=${1:-"churro_htr_htromance_exp1"}
EPOCHS=${2:-5}
TRAIN_BATCH_SIZE=${3:-4}
GRAD_ACCUM=${4:-1}
LORA_R=${5:-8}
LEARNING_RATE=${6:-0.0001}
WEIGHT_DECAY=${7:-0.01}
MAX_NEW_TOKENS=${8:-700}
LORA_DROPOUT=${9:-0}

# Fixed parameters
MODEL_NAME="stanford-oval/churro-3B"
DATA_TRAIN="../data/HTRomance-french/data/train"
DATA_VALID="../data/HTRomance-french/data/valid"
DATA_TEST="../data/HTRomance-french/data/test"
OUTPUT_DIR="results"
DEVICE="cuda"
USE_WANDB=true
WANDB_PROJECT="HTR-comparison"

# Paths
CONFIG_DIR="configs/experiments"
CONFIG_FILE="${CONFIG_DIR}/${EXPERIMENT_NAME}.yml"
CONFIG_RESULT="src/tasks/htr/models/${EXPERIMENT_NAME}-finetuned/inference_config.yml"

# ========================================
# BANNER
# ========================================

echo "========================================"
echo "  VLM Training & Evaluation Pipeline"
echo "========================================"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Epochs: ${EPOCHS}"
echo "Batch Size: ${TRAIN_BATCH_SIZE}"
echo "Gradient Accumulation: ${GRAD_ACCUM}"
echo "LoRA R: ${LORA_R}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "========================================"
echo ""

# ========================================
# STEP 0: CREATE CONFIG DIRECTORY
# ========================================

mkdir -p ${CONFIG_DIR}

# ========================================
# STEP 1: GENERATE CONFIG FILE
# ========================================

echo "[1/4] Generating config file: ${CONFIG_FILE}"

cat > ${CONFIG_FILE} << EOF
run_name: "${EXPERIMENT_NAME}"
output_dir: "${OUTPUT_DIR}"
device: "${DEVICE}"
use_wandb: ${USE_WANDB}
wandb_project: "${WANDB_PROJECT}"

data:
  train: "${DATA_TRAIN}"
  valid: "${DATA_VALID}"
  test: "${DATA_TEST}"

tasks:
  htr:
    type: VLMHTR
    config:
      model_name: '${MODEL_NAME}'
      max_new_tokens: ${MAX_NEW_TOKENS}
      batch_size: 10
      use_4bit: true
      lora_r: ${LORA_R}
      lora_dropout: ${LORA_DROPOUT}
      max_seq_length: 4096
      train_batch_size: ${TRAIN_BATCH_SIZE}
      gradient_accumulation_steps: ${GRAD_ACCUM}
      warmup_ratio: 0.1
      epochs: ${EPOCHS}
      learning_rate: ${LEARNING_RATE}
      weight_decay: ${WEIGHT_DECAY}
      dataset_num_proc: 4
      prompt: > 
        You are a paleographer specializing in medieval languages.
        Follow these instructions:

        1. You will be provided with a scanned document page.

        2. Perform transcription on the main text from this image, line by line, from top to bottom.

        3. If you encounter any non-text elements, simply skip them without attempting to describe them.

        4. Do not translate, modernize or standardize the text. 
        For example, if the transcription is using "ſ" instead of "s" or "а" instead of "a", keep it that way.

        5. Do not include any other words or separator in your response.

        Remember, your goal is to accurately transcribe the text from the scanned page as much as possible. 
        Process the entire page, even if it contains a large amount of text, and provide clear, well-formatted output. 
        Pay attention to the appropriate reading order and layout of the text.

EOF

echo "✓ Config file created"
echo ""

# ========================================
# STEP 2: TRAINING
# ========================================

echo "[2/4] Starting training..."

# Train the model
docworkflow -c ${CONFIG_FILE} train -t htr

echo "✓ Training completed"
echo ""

# ========================================
# STEP 3: PREDICTION
# ========================================

echo "[3/4] Starting prediction on test set..."

# Predict on test set
docworkflow -c ${CONFIG_RESULT} predict -t htr -d test

echo "✓ Prediction completed"
echo ""

# ========================================
# STEP 4: EVALUATION
# ========================================

echo "[4/4] Starting evaluation..."

# Evaluate predictions
docworkflow -c ${CONFIG_RESULT} score -t htr -d test

echo "✓ Evaluation completed"
echo ""

# ========================================
# SUMMARY
# ========================================

echo "========================================"
echo "  Pipeline Complete!"
echo "========================================"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Results saved in: ${OUTPUT_DIR}/${EXPERIMENT_NAME}"
echo "Config file: ${CONFIG_FILE}"
echo "========================================"