# Document Analysis Framework

A comprehensive framework for automatic document analysis focusing on two tasks:
- **Layout segmentation**: Detection of Segmonto zone types in document images
- **Line segmentation**: Detection of text lines within document zones

## Features

- Modular and object-oriented architecture
- Unified command line interface for both layout and line segmentation
- Multiple model support:
  - YOLO-based models for layout segmentation
  - Kraken/YALTAi-based models for line segmentation
- ALTO XML format for data representation and interchange
- Integration with Weights & Biases for experiment tracking
- Comprehensive visualization tools for analysis results
- Centralized configuration management with JSON files
- Performance evaluation metrics (mAP, precision, recall)

## Installation

```bash
git clone https://github.com/your-username/document-analysis.git
cd document-analysis
pip install -r requirements.txt
```

## Usage

### Layout Segmentation

```bash
# Evaluate a layout segmentation model
python run.py layout --function eval --configs layout/configs/catmus_seg_s_16_10.json

# Train a layout segmentation model
python run.py layout --function train --configs layout/configs/catmus_seg_s_16_10.json

# Generate predictions with a layout model and save ALTO XML files
python run.py layout --function predict --configs layout/configs/catmus_seg_s_16_10.json --output /path/to/output/dir

# Visualize layout segmentation results
python run.py layout --function print --corpus_path /path/to/images --output /path/to/output/dir
```

### Line Segmentation

```bash
# Evaluate a line segmentation model
python run.py line --function eval --configs line/configs/catmus_ms_line.json

# Train a line segmentation model
python run.py line --function train --configs line/configs/catmus_ms_line.json

# Generate predictions with a line model
python run.py line --function predict --configs line/configs/catmus_ms_line.json --output /path/to/output/dir

# Visualize line segmentation results
python run.py line --function print --corpus_path /path/to/images --output /path/to/output/dir
```

## Configuration Files

Model configurations are stored in JSON files. Here are examples for each type:

### Layout Segmentation Configuration

```json
{
    "name": "catmus_seg_s_16_10",
    "model_path": "layout/LA-training/catmus_seg_s_16_10/weights/best.pt",
    "data_path": "../data_yolo/medieval-segmentation-yolo/config.yaml",
    "corpus_path": false,
    "data": "catmus_seg",
    "training_mode": "restricted",
    "batch_size": 16,
    "epochs": 10,
    "pretrained_w": "layout/models/yolo11s.pt",
    "img_size": 640,
    "use_wandb": false
}
```

### Line Segmentation Configuration

```json
{
    "name": "catmus_ms_line",
    "data_path": "../data/medieval-segmentation/src/altos/test",
    "model_path": "models/line/baseline_ms_medieval.mlmodel",
    "corpus_path": false,
    "data": "catmus_lines",
    "training_mode": "restricted",
    "batch_size": 16,
    "epochs": 100,
    "img_size": 1024,
    "text_direction": "horizontal-lr",
    "iou_threshold": 0.5,
    "buffer_size": 5,
    "use_wandb": true
}
```

## Command Line Arguments

The framework supports the following command line arguments:

- `--function`: Function to run (`eval`, `train`, `predict`, `print`)
- `--data_path`: Path to training/validation/test data (overrides config file)
- `--corpus_path`: Path to corpus data for additional testing or visualization
- `--pred_path`: Path to prediction files (only for `predict` function)
- `--xml_path`: Path to ALTO XML files if different from image path
- `--configs`: Path to JSON configuration file(s)
- `--output`: Path to save evaluation results or visualizations


## Evaluation Metrics

The framework calculates and reports the following metrics:
- mAP (mean Average Precision) at different IoU thresholds (0.5-0.95, 0.5, 0.75)
- Precision
- Recall
- Specific metrics for MainZone detection

## Architecture

The framework follows a modular architecture:
- `core/`: Core functionality and utilities
- `layout/`: Layout segmentation models and configurations
- `line/`: Line segmentation models and configurations
- `run.py`: Main entry point

## Dependencies

The framework relies on several key libraries:
- YOLO/Ultralytics for layout detection
- Kraken/YALTAi for line detection
- PyTorch for deep learning
- Weights & Biases for experiment tracking
- Matplotlib for visualization

## Credits

This project uses several work done by Thibault Clérice:
- [YALTAi](https://github.com/ponteineptique/YALTAi) that provide function to adapt YOLO file for Kraken
- [RTK](https://github.com/ponteineptique/rtk) for the task management library inspiration :)
- [yolalto](https://github.com/ponteineptique/yolalto) for converting YOLO predictions to ALTO XML format
