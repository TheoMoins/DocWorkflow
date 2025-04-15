# Document Analysis Framework

This framework is designed for automatic document analysis, currently focusing on two tasks:
- **Layout segmentation**: detection of Segmonto zone types in a document
- **Line segmentation**: detection of text lines within a document

## Features

- Modular and object-oriented architecture
- Unified command line interfaces
- Support for different layout and line segmentation models
- Integration with Weights & Biases for experiment tracking
- Centralized configuration management

## Installation

```bash
git clone https://github.com/your-username/document-analysis.git
cd document-analysis
pip install -r requirements.txt
```

## Usage

### Layout Segmentation

```bash
# Evaluation with a specific model
python run.py layout --function eval --config /path/to/config.json

# Training with a specific model
python run.py layout --function train --config /path/to/config.json
```

### Line Segmentation

```bash
# Evaluation with a specific model
python run.py line --function eval --config /path/to/config.json

# Training with a specific model
python run.py line --function train --config /path/to/config.json
```

## Configurations

Model configurations are stored in JSON files. Here's an example for a layout segmentation model:

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
    "pretrained_w": "yolo11s",
    "img_size": 640,
    "use_wandb": false
}
```
