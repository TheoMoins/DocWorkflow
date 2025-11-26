#!/bin/bash

# Script: run_hyperparameter_sweep.sh
# Run multiple experiments with different hyperparameters

set -e

echo "========================================"
echo "  Hyperparameter Sweep"
echo "========================================"
echo ""

# Make the main script executable
chmod +x train_predict_eval_churro.sh

# ========================================
# EXPERIMENT 1: Baseline (5 epochs)
# ========================================

echo "Running Experiment 1: Baseline (5 epochs)"
./train_predict_eval_churro.sh \
    "churro_5ep_bs4_lr1e4" \
    5 \
    4 \
    1 \
    8 \
    0.0001

# ========================================
# EXPERIMENT 2: More epochs (10)
# ========================================

echo "Running Experiment 2: 10 epochs"
./train_predict_eval_churro.sh \
    "churro_10ep_bs4_lr1e4" \
    10 \
    4 \
    1 \
    8 \
    0.0001

# ========================================
# EXPERIMENT 3: Even more epochs (20)
# ========================================

echo "Running Experiment 3: 20 epochs"
./train_predict_eval_churro.sh \
    "churro_20ep_bs4_lr1e4" \
    20 \
    4 \
    1 \
    8 \
    0.0001

# ========================================
# EXPERIMENT 4: Larger batch with gradient accumulation
# ========================================

echo "Running Experiment 4: Larger effective batch (bs=2, grad_accum=4)"
./train_predict_eval_churro.sh \
    "churro_10ep_bs2_ga4_lr1e4" \
    10 \
    2 \
    4 \
    8 \
    0.0001

# ========================================
# EXPERIMENT 5: Higher learning rate
# ========================================

echo "Running Experiment 5: Higher learning rate (2e-4)"
./train_predict_eval_churro.sh \
    "churro_10ep_bs4_lr2e4" \
    10 \
    4 \
    1 \
    8 \
    0.0002

# ========================================
# EXPERIMENT 6: Lower learning rate
# ========================================

echo "Running Experiment 6: Lower learning rate (5e-5)"
./train_predict_eval_churro.sh \
    "churro_10ep_bs4_lr5e5" \
    10 \
    4 \
    1 \
    8 \
    0.00005

# ========================================
# EXPERIMENT 7: Higher LoRA rank
# ========================================

echo "Running Experiment 7: Higher LoRA rank (r=16)"
./train_predict_eval_churro.sh \
    "churro_10ep_bs4_lora16" \
    10 \
    4 \
    1 \
    16 \
    0.0001

# ========================================
# EXPERIMENT 8: Long training (50 epochs)
# ========================================

echo "Running Experiment 8: Long training (50 epochs)"
./train_predict_eval_churro.sh \
    "churro_50ep_bs4_lr1e4" \
    50 \
    4 \
    1 \
    8 \
    0.0001

# ========================================
# SUMMARY
# ========================================

echo ""
echo "========================================"
echo "  All Experiments Complete!"
echo "========================================"
echo "Results are in the 'results/' directory"
echo "Check WandB for detailed metrics"
echo "========================================"