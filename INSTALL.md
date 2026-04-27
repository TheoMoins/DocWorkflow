# DocWorkflow — Installation & Environment Guide

DocWorkflow uses **pixi** to manage three isolated environments:

| Environment | Purpose | Key packages |
|---|---|---|
| `main` | Layout segmentation, line detection, scoring — CPU only | kraken, yaltai, ultralytics, peft |
| `inference` | VLM HTR inference — requires a CUDA GPU | same as `main` + CUDA-enabled pytorch |
| `train` | VLM fine-tuning — requires a CUDA GPU | unsloth, trl, peft, bitsandbytes |

A third sub-project (`churro/`) has its own independent pixi workspace and is not covered here.

---

## Prerequisites

### 1. Install pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Verify: `pixi --version` (tested with 0.56.0+).

### 2. CUDA (for `inference` and `train` only)

The `inference` and `train` environments require a CUDA 12.x driver (12.1+ recommended).
The CUDA toolkit itself is bundled via the `pytorch` conda channel — no manual CUDA setup needed.

---

## Installation

From the repository root:

```bash
# CPU-only tasks (layout, scoring, text extraction, etc.)
pixi install -e main
pixi run install-yaltai   # post-install step: yaltai cannot be resolved by pixi directly

# VLM HTR inference on a GPU server
pixi install -e inference
pixi run -e inference install-yaltai

# VLM fine-tuning (always needs a GPU)
pixi install -e train
```

`pixi.lock` is committed to the repository — `pixi install` is therefore deterministic across machines.

---

## Activating environments

```bash
# Interactive shell with main env
pixi shell -e main

# Interactive shell with inference env (GPU server)
pixi shell -e inference

# Interactive shell with train env
pixi shell -e train

# Exit back to your normal shell
exit
```

> **Legacy virtualenvs** (`envs/main`, `envs/vlm-training`) still exist and can be activated with `source envs/main/bin/activate`. They are no longer the recommended path — use pixi instead.

---

## Running commands

### CLI entry point

```bash
# Layout / line detection / scoring (CPU)
pixi run -e main docworkflow -c configs/example_config.yml predict -t layout -d test -o results/
pixi run -e main docworkflow -c configs/example_config.yml score -t all -d test -p results/

# VLM HTR inference (GPU server)
pixi run -e inference docworkflow -c configs/example_config.yml predict -t htr -d test -o results/

# Training
pixi run -e train docworkflow -c configs/example_config.yml train -t htr
```

### Tests

```bash
# Run all tests (CPU)
pixi run -e main pytest tests/ -v

# Skip tests that require a GPU
pixi run -e main pytest tests/ -v -m "not requires_gpu and not requires_training"

# Training smoke tests (requires a GPU)
pixi run -e train pytest tests/test_imports_train.py -v
```

---

## Configuration

All CLI commands take a `-c / --config` YAML file. See `configs/example_config.yml` for a fully annotated example.

Key fields:

```yaml
run_name: "experiment_01"
output_dir: "results"
device: "cuda"         # or "cpu" — auto-detected if omitted
use_wandb: false

data:
  train: "path/to/train/data"
  test:  "path/to/test/data"

tasks:
  layout:
    type: YoloLayout
    config:
      model_path: "path/to/weights.pt"
      pretrained_w: "yolo11s.pt"
      batch_size: 16
      img_size: 640
      epochs: 50

  line:
    type: KrakenLine
    config:
      model_path: "path/to/baseline.mlmodel"
      text_direction: "horizontal-lr"

  htr:
    type: VLMLineHTR
    config:
      model_name: "path/to/model"
      max_new_tokens: 128
      line_batch_size: 4
```

Tasks are optional: include only the ones you need.

---

## Test markers

| Marker | Meaning |
|---|---|
| `requires_main` | needs kraken, yaltai, ultralytics |
| `requires_training` | needs unsloth, peft, trl, datasets; skips if no GPU |
| `requires_gpu` | needs CUDA |

---

## Updating the lock file

After adding or changing a dependency in `pixi.toml`:

```bash
pixi install -e main
pixi install -e inference
pixi install -e train
git add pixi.lock
git commit -m "update pixi.lock"
```

> Do not edit `pixi.lock` by hand.

---

## Troubleshooting

**`yaltai` or `fast-deskew` not found after install**  
Run `pixi run install-yaltai` (or `pixi run -e inference install-yaltai`). This post-install step is not automatic.

**`torch.cuda.is_available()` returns `False` on a GPU server**  
Make sure you are using the `inference` environment, not `main`. The `main` env installs a CPU-only pytorch build from conda-forge by design.

**`pixi install -e inference` fails with channel conflicts**  
The workspace uses `channel-priority = "disabled"` to allow pytorch, nvidia, and conda-forge to coexist. If you see a conflict after changing `pixi.toml`, verify that this setting is present in `[workspace]`.

**`unsloth` raises `NotImplementedError: Unsloth cannot find any torch accelerator`**  
Expected on machines without a GPU. Training tests skip automatically in this case.

**`fsspec` extra `http` warning during `pixi install -e train`**  
Harmless. Recent fsspec versions integrated HTTP support into the base package; the extra is silently ignored.
