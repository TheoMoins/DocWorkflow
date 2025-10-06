# DocWorkflow

**Document Analysis Framework** - A modular pipeline for document layout analysis, line segmentation, and handwritten text recognition (HTR).

## Overview

DocWorkflow is a Python framework designed for end-to-end document analysis workflows. It provides a unified interface for training, evaluating, and running inference on three key document analysis tasks:

- **Layout Segmentation**: Detect and classify document regions (text blocks, margins, illustrations, etc.)
- **Line Segmentation**: Extract text lines with baselines from document regions
- **Handwritten Text Recognition (HTR)**: Transcribe text from segmented lines

The framework integrates YOLO for layout, Kraken for lines and HTR and uses ALTO XML as the standard format for annotations and predictions.


## Quick Start

### 1. Configure Your Workflow

Create a configuration file (e.g., `config.yml`):

```yaml
# Global parameters
run_name: "my_experiment"
output_dir: "results"
device: "cuda"  # or "cpu"
use_wandb: false

# Data paths
data:
  train: "path/to/train/data"
  valid: "path/to/valid/data"
  test: "path/to/test/data"

# Tasks configuration
tasks:
  layout: 
    type: YoloLayout
    config:
      model_path: "path/to/layout_model.pt"
      pretrained_w: "path/to/yolo11s.pt"  # For training
      batch_size: 16
      img_size: 640
      epochs: 50

  line:
    type: KrakenLine
    config:
      model_path: "path/to/baseline_model.mlmodel"
      text_direction: "horizontal-lr"

  htr:
    type: KrakenHTR
    config:
      model_path: "path/to/htr_model.mlmodel"
```

### 2. Run Prediction

Predict on a dataset:

```bash
docworkflow -c config.yml predict -t layout -d test
```

### 3. Evaluate Results

Score predictions against ground truth:

```bash
# Score with custom prediction path
docworkflow -c config.yml score -t layout -d test -p results/layout/
```

### 4. Train Models

Train a model on your dataset:

```bash
# Train with custom seed
docworkflow -c config.yml train -t layout -s 42
```

### 5. Visualize Results

Generate visual outputs:

```bash
# Visualize layout segmentation
docworkflow -c config.yml print -t layout -p results/layout/ -o viz/

# Visualize line segmentation
docworkflow -c config.yml print -t line -p results/line/ -o viz/
```


## Data Format

DocWorkflow uses **ALTO XML** as the primary format for annotations and predictions. ALTO is a standard XML schema for describing the layout and content of physical text resources.

### Input Data Structure

```
dataset/
├── image-001.jpg
├── image-001.xml    # ALTO XML with annotations
├── image-002.jpg
├── image-002.xml
└── ...
```

## Evaluation Metrics

### Layout & Line Segmentation
- **mAP@50-95**: Mean Average Precision across IoU thresholds
- **mAP@50**: Mean Average Precision at IoU=0.5
- **mAP@75**: Mean Average Precision at IoU=0.75
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

### HTR (Coming Soon)
- **CER**: Character Error Rate
- **WER**: Word Error Rate

## Advanced Usage

### Custom Models

Add your own model by:

1. Creating a new task class in `src/tasks/`
2. Inheriting from `BaseTask`
3. Implementing required methods: `load()`, `train()`, `predict()`, `score()`
4. Registering in `src/cli/config/constants.py`

### Pipeline Chaining

Use the output of one task as input for the next:

```bash
# 1. Layout segmentation
docworkflow -c config.yml predict -t layout -d test -o results/step1/

# 2. Line segmentation (uses layout from step 1)
docworkflow -c config.yml predict -t line -d results/step1/ -o results/step2/

# 3. HTR (uses lines from step 2)
docworkflow -c config.yml predict -t htr -d results/step2/ -o results/final/
```

### Pre-computed Inputs

If you already have layout or line segmentation, specify `input_file` in config:

```yaml
tasks:
  line:
    type: KrakenLine
    config:
      input_file: "path/to/precomputed/layout/"
```

## Acknowledgments

This framework builds upon:
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for layout segmentation
- [Kraken](https://github.com/mittagessen/kraken) for line segmentation and HTR
- [YALTAi](https://github.com/PonteIneptique/YALTAi) for ALTO utilities

