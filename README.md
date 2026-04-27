# DocWorkflow

**Document Analysis Framework** - A modular pipeline for document layout analysis, line segmentation, and handwritten text recognition (HTR).

## Overview

DocWorkflow is a Python framework designed for end-to-end document analysis workflows. It provides a unified interface for training, evaluating, and running inference on three key document analysis tasks:

- **Layout Segmentation**: Detect and classify document regions (text blocks, margins, illustrations, etc.)
- **Line Segmentation**: Extract text lines with baselines from document regions
- **Handwritten Text Recognition (HTR)**: Transcribe text from segmented lines

The framework integrates YOLO for layout and line detection, Kraken and Vision Language Models (VLMs) for HTR, and uses ALTO XML as the standard format for annotations and predictions.


## Installation

DocWorkflow uses **[pixi](https://pixi.sh)** to manage three isolated environments:

| Environment | Purpose | GPU required |
|---|---|---|
| `main` | Layout, line detection, scoring — CPU only | No |
| `inference` | VLM HTR inference | Yes (CUDA 12.x) |
| `train` | VLM fine-tuning (LoRA) | Yes (CUDA 12.x) |

### 1. Install pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### 2. Install environments

```bash
# CPU tasks (layout, lines, scoring, Kraken HTR)
pixi install -e main
pixi run install-yaltai   # required post-install step

# VLM HTR inference (GPU server)
pixi install -e inference
pixi run -e inference install-yaltai

# VLM fine-tuning
pixi install -e train
```

`pixi.lock` is committed — `pixi install` is deterministic across machines.

### 3. Activate an environment

All commands below assume the environment is already active. Activate it once per session:

```bash
pixi shell -e main        # CPU tasks (layout, lines, scoring)
pixi shell -e inference   # VLM HTR inference (GPU)
pixi shell -e train       # VLM fine-tuning (GPU)
exit                      # back to your normal shell
```

## Quick Start

### 1. Configure Your Workflow

Create a configuration file (e.g., `config.yml`). See `configs/example_config.yml` for a fully annotated example.

```yaml
run_name: "my_experiment"
output_dir: "results"
device: "cuda"       # or "cpu" — auto-detected if omitted
use_wandb: false

data:
  train: "path/to/train/data"
  valid: "path/to/valid/data"
  test: "path/to/test/data"

tasks:
  layout:
    type: YoloLayout
    config:
      model_path: "path/to/layout_model.pt"
      pretrained_w: "path/to/yolo11s.pt"   # for training only
      batch_size: 16
      img_size: 640
      epochs: 50

  line:
    type: KrakenLine
    config:
      model_path: "path/to/baseline_model.mlmodel"
      text_direction: "horizontal-lr"

  htr:
    type: VLMLineHTR
    config:
      model_name: "path/to/htr_model"
      max_new_tokens: 128
      line_batch_size: 4
```

Tasks are optional — include only the ones you need.

### 2. Run the Full Pipeline

Activate the environment once, then run commands directly:

```bash
pixi shell -e main
```

Run all three tasks sequentially (layout → line → HTR):

```bash
docworkflow -c config.yml predict -t all
```

Or run individual tasks:

```bash
docworkflow -c config.yml predict -t layout -d test
```

### 3. Evaluate Results

```bash
docworkflow -c config.yml score -t all -d test
```

### 4. Train a Model

```bash
pixi shell -e train
docworkflow -c config.yml train -t htr
```

### 5. Visualize Results

```bash
# Visualize layout or line segmentation
docworkflow -c config.yml print -t layout -p results/layout/ -o viz/

# Export HTR results as JSON (competition format)
docworkflow -c config.yml print -t htr -p results/htr/ -o viz/ --json
```


## CLI Reference

**Global option:**

```
docworkflow -c <config.yml> <command> [options]
```

### `predict`

Run inference on a dataset.

```
-t, --task    [layout|line|htr|all]    Task to run (default: all)
-d, --dataset [train|valid|test]       Dataset split (default: test)
-o, --output  <path>                   Output directory
--save_image                           Force saving images in output
--no_save_image                        Force NOT saving images in output
--cleanup_intermediate                 Delete intermediate outputs when using -t all
```

When `-t all` is used, the pipeline runs layout → line → HTR automatically, passing the output of each step as input to the next.

### `score`

Evaluate predictions against ground truth.

```
-t, --task    [layout|line|htr|all]    Task to score (default: all)
-d, --dataset [train|valid|test]       Ground truth split (default: test)
-p, --pred_path <path>                 Predictions directory (optional)
-o, --output  <path>                   Output directory for CSV results
```

Produces: `results.csv`, `scores_per_page.csv`, `scores_per_document.csv` (hierarchical datasets).

### `train`

Train a model.

```
-t, --task [layout|line|htr]    Task to train (required)
-s, --seed <int>                Random seed (default: 42)
```

### `print`

Generate visualizations and exports.

```
-t, --task    [layout|line|htr]    Task to visualize (default: htr)
-p, --pred_path <path>             ALTO XML directory (required)
-o, --output  <path>               Output directory
--json                             Export HTR results as submission.json
```

### `prepare-yolo-lines`

Convert ALTO XML line annotations to YOLO format (utility command).

```
-i, --input  <path>    Input directory with ALTO XMLs
-o, --output <path>    Output directory (YOLO format)
--polygon              Generate segmentation polygons instead of bounding boxes
```

Preserves `train/` `val/` `test/` subdirectory structure.


## Configuration Reference

### Global parameters

```yaml
run_name: <string>           # Experiment name (used in output paths and W&B)
output_dir: <string>         # Base output directory (default: results/)
device: [cuda|cpu]           # Compute device (auto-detected if omitted)
use_wandb: <bool>            # Enable W&B logging (default: false)
wandb_project: <string>      # W&B project name
save_image: <bool>           # Copy source images to output (default: true)
use_metadata: <bool>         # Load metadata.json for aggregated stats (default: false)
reading_order: <string>      # Line ordering algorithm: "dbscan" (default)
```

### Task types

Each task entry follows the pattern:

```yaml
tasks:
  <layout|line|htr>:
    type: <ClassName>
    config:
      ...
```

#### Layout

| Type | Backend | Training |
|------|---------|----------|
| `YoloLayout` | Ultralytics YOLO | Yes |

```yaml
type: YoloLayout
config:
  model_path: <path>        # Trained weights (.pt)
  pretrained_w: <path>      # YOLO pretrained weights (training only)
  batch_size: <int>
  img_size: <int>           # default: 640
  epochs: <int>
  input_file: <path>        # Use pre-computed layout instead of running the model
```

#### Line segmentation

| Type | Backend | Training |
|------|---------|----------|
| `KrakenLine` | Kraken (via YALTAi) | No |
| `YoloLine` | Ultralytics YOLO | Yes |

```yaml
type: KrakenLine          # or YoloLine
config:
  model_path: <path>
  text_direction: <string>  # e.g. "horizontal-lr" (KrakenLine)
  pretrained_w: <path>      # YOLO only
  batch_size: <int>         # YOLO only
  img_size: <int>           # YOLO only
  epochs: <int>             # YOLO only
  input_file: <path>        # Use pre-computed layout
```

#### HTR

| Type | Backend | Training | Notes |
|------|---------|----------|-------|
| `KrakenHTR` | Kraken | No | |
| `TrOCRHTR` | HuggingFace TrOCR | No | |
| `VLMLineHTR` | Vision LM, line-level | Yes (LoRA) | Recommended |
| `VLMMultiLineHTR` | Vision LM, multi-line | Yes (LoRA) | |
| `VLMPageHTR` | Vision LM, page-level | No | |
| `VLMLineHTRSilver` | Vision LM, text-only | Yes | Pre-training without images |

**KrakenHTR / TrOCRHTR:**

```yaml
type: KrakenHTR           # or TrOCRHTR
config:
  model_path: <path>
  max_length: <int>       # TrOCR only
  num_beams: <int>        # TrOCR only
  input_file: <path>      # Use pre-computed lines
```

**VLMLineHTR — inference:**

```yaml
type: VLMLineHTR
config:
  model_name: <path|hf-id>      # Local path or HuggingFace model ID
  max_new_tokens: <int>         # default: 512
  line_batch_size: <int>        # default: 1
  prompt: <string>              # Custom prompt; use {conventions} as placeholder
  max_pixels: <int>             # default: 401408 (512×28×28)
  device_map: [auto|cpu|cuda]
  attn_implementation: <string> # e.g. flash_attention_2
  use_4bit: <bool>              # 4-bit quantization
  use_8bit: <bool>              # 8-bit quantization
  base_model: <path|hf-id>      # Base model when model_name is a LoRA adapter
  input_file: <path>            # Use pre-computed lines
```

**VLMLineHTR — training:**

```yaml
type: VLMLineHTR
config:
  model_name: <path|hf-id>
  base_model: <path|hf-id>      # Base model for LoRA
  model_dir: <path>             # Where to save the adapter

  # LoRA
  lora_r: <int>                 # default: 16
  lora_dropout: <float>         # default: 0
  use_rslora: <bool>            # Rank-stabilized LoRA (default: false)

  # Training
  max_seq_length: <int>         # default: 1024
  train_batch_size: <int>       # default: 4
  gradient_accumulation_steps: <int>  # default: 4
  epochs: <int>                 # default: 3
  learning_rate: <float>        # default: 2e-4
  weight_decay: <float>         # default: 0.01
  warmup_ratio: <float>         # default: 0.1
  special_char_weighting: <float|null>
```

Supported VLM base models: Qwen2.5-VL, Qwen3-VL, IDEFICS3, MiniCPM-V.


## Data Format

DocWorkflow uses **ALTO XML** as the primary format for annotations and predictions.

### Input data structure

```
dataset/
├── image-001.jpg
├── image-001.xml    # ALTO XML with annotations
├── image-002.jpg
├── image-002.xml
└── ...
```

Hierarchical structures (subdirectories per document) are automatically detected and preserved in outputs.

### Pipeline data flow

```
images + ALTO XML (layout regions)
    → YoloLayout / KrakenLine → ALTO XML (text lines)
    → KrakenHTR / VLMLineHTR → ALTO XML (transcribed text)
```


## Evaluation Metrics

### Layout & Line segmentation

- **mAP@50-95**: Mean Average Precision across IoU thresholds
- **mAP@50**: Mean Average Precision at IoU=0.5
- **mAP@75**: Mean Average Precision at IoU=0.75
- **Precision** / **Recall** at IoU=0.75

### HTR

- **CER**: Character Error Rate (with NFD normalization for diacritics)
- **WER**: Word Error Rate
- **Substitutions / Insertions / Deletions**: Detailed error counts
- **Worst pages**: Top 5 pages by CER

Results are saved as CSV files. When `use_wandb: true`, per-page and per-document tables are uploaded as W&B artifacts.


## Advanced Usage

### Pipeline chaining (manual)

Use the output of one task as input to the next with `input_file`:

```yaml
tasks:
  line:
    type: KrakenLine
    config:
      input_file: "path/to/precomputed/layout/"
```

Or pass the output directory directly with `-o` / `-p`:

```bash
docworkflow -c config.yml predict -t layout -d test -o results/step1/
docworkflow -c config.yml predict -t line   -d test -p results/step1/ -o results/step2/
docworkflow -c config.yml predict -t htr    -d test -p results/step2/ -o results/final/
```

### Adding a custom model

1. Create a new task class in `src/tasks/`
2. Inherit from `BaseTask` (or `BaseLine` / `BaseHTR`)
3. Implement required methods: `load()`, `train()`, `_process_batch()`, `_score_batch()`
4. Add an entry to the `ModelImports` enum in `src/cli/config/constants.py`


## Acknowledgments

This framework builds upon:
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for layout and line segmentation
- [Kraken](https://github.com/mittagessen/kraken) for line segmentation and HTR
- [YALTAi](https://github.com/PonteIneptique/YALTAi) for ALTO/YOLO utilities
- [Unsloth](https://github.com/unslothai/unsloth) for efficient VLM fine-tuning
