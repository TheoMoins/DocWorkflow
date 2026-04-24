# Known Issues — Dependency Management

## Context

The project uses two virtualenvs managed by `scripts/setup_envs.sh`:

- `envs/main` (Python 3.10) — layout segmentation, HTR inference (yaltai, kraken, ultralytics)
- `envs/vlm-training` (Python 3.10) — fine-tuning VLMs (unsloth, trl, peft, bitsandbytes)

A third environment (`churro/`) is managed independently with pixi and works correctly.

---

## Root Cause: Wrong-Level Dependencies

Both conflicts share the same structural problem: a **heavy library is pulled in for a small
subset of its functionality**, and that library's transitive dependencies conflict with the
rest of the environment.

### `kraken` in `vlm-training` — ML library used only as an XML parser

`kraken` is a full HTR framework (hundreds of MB, strict torch version pins). In the
`vlm-training` environment it is needed for **one thing only**: `kraken.lib.xml.XMLPage`,
used in `src/alto/alto_lines.py:read_lines_geometry()` to parse ALTO XML files.

`XMLPage` gives access to `.imagename`, `.regions`, `.lines`, and per-line `.baseline`,
`.boundary`, `.text`. Everything else in `alto_lines.py` is pure `lxml`. This is a
**zero-ML operation** — parsing a structured XML file — wrapped inside a library that
requires a specific torch ecosystem. kraken's torch constraints are incompatible with
unsloth's recent-torch requirements, hence the conflict.

**Fix**: Replace `XMLPage` with a lightweight `lxml`-based parser in `src/alto/alto_lines.py`.
This removes the kraken dependency from `vlm-training` entirely. kraken stays in `main`
where it is genuinely needed for `KrakenLineTask` and `KrakenHTRTask` model inference.

### `yaltai` in `main` — segmentation wrapper that pins old torch

`yaltai` wraps kraken's baseline segmentation with a YOLO layout step. It is used in
`src/tasks/line/kraken_line.py` only, and pins specific kraken + torch versions. This
creates a downward pressure on the torch version in `main`, which conflicts with any
dependency that requires a more recent torch (e.g. newer ultralytics, newer transformers
for VLM inference). In `setup_envs.sh`, yaltai is installed with `--no-deps` precisely
to avoid this.

**Fix**: In the future `pixi.toml`, declare yaltai with `no-deps = true` (as already
templated) and explicitly list only the transitive deps it actually needs. This isolates
the torch constraint to the model-inference task rather than infecting the whole env.

---

## Known Conflicts

### 1. `vlm-training`: pyproject.toml constraints are systematically violated

`setup_envs.sh` installs the package in vlm-training with `pip install -e . --no-deps` precisely
because the constraints in `pyproject.toml` are incompatible with the packages required by unsloth.

Concrete violations (as of April 2025 wandb snapshot):

| Dependency | pyproject.toml constraint | Actually installed |
|---|---|---|
| `torch` | `>=2.1.0,<2.5.0` | `2.10.0` |
| `torchvision` | `>=0.19.0,<0.24.0` | `0.25.0` |
| `kraken` | `>=5.3.0,<6.0.0` | `6.0.3` |
| `pandas` | `>=2.0.0,<2.3.0` | `2.3.3` |
| `matplotlib` | `>=3.7.0,<3.9.0` | `3.10.8` |
| `transformers` | `>=4.55.0` | `5.2.0` |

`--no-deps` silences the resolver but does not fix the conflict: the installed environment
is not described by `pyproject.toml` and is therefore not reproducible from it.

### 2. `torch` / `torchvision` version coupling is under-constrained

The constraints `torch>=2.1.0,<2.5.0` and `torchvision>=0.19.0,<0.24.0` look broad but are
actually almost point-locked: torchvision 0.19.x requires exactly torch 2.4.x, and
torchvision 0.20.x+ requires torch 2.5+, which violates the upper bound. The only valid
combination is `torch==2.4.x + torchvision==0.19.x`. Every other combination will silently
install mismatched versions unless a constraints file enforces them jointly.

### 3. `unsloth` and recent `torch` are incompatible with the `main` environment

unsloth ≥ 2025.x requires torch ≥ 2.5, while the `main` environment targets torch ~2.4
(for ultralytics + kraken 5.x compatibility). These two ecosystems cannot share a single
`[dependencies]` block in `pyproject.toml`.

### 4. `setup_envs.sh` uses `--no-deps` as a workaround in multiple places

```bash
pip install --no-deps yaltai          # missing transitive deps
pip install --no-deps trl peft accelerate bitsandbytes
pip install --no-deps rich iso639
pip install -e . --no-deps            # ignores all pyproject.toml constraints
```

Each `--no-deps` call is a sign that the resolver cannot satisfy the declared requirements.
The result is an environment that may work on the day it is built but will break after any
`pip install --upgrade` or on a fresh machine with different pre-installed packages.

### 5. No lock files → non-reproducible environments

`pip install package` without a lock resolves to the latest compatible version at install
time. Two machines running `setup_envs.sh` one month apart will produce different
environments.

---

## Root Cause

`pyproject.toml` declares a single dependency set that covers both the inference stack
(kraken, ultralytics, torch ~2.4) and the training stack (unsloth, recent torch, recent
transformers). These two stacks are currently incompatible with each other at the version
level. The `--no-deps` flags and the separate virtualenv are workarounds for a structural
problem: **one `pyproject.toml` cannot describe two mutually incompatible environments**.

---

## Recommended Solution: pixi with per-environment feature groups

The `churro/` sub-project already demonstrates the correct approach with `pixi.toml`.
Apply the same pattern to the root project.

### Why pixi

- `pixi.lock` provides a complete, cross-platform dependency lock (exact versions + hashes)
- Feature groups (`[feature.train]`, `[feature.main]`) allow declaring incompatible stacks
  in a single manifest without mixing them
- `pixi install` is idempotent and reproducible; `pixi run` activates the correct env
  automatically
- conda-forge packages (e.g. CUDA libraries, torch with CUDA) are resolved jointly,
  eliminating the PyPI-only torchvision/torch coupling issue

### Migration steps

1. **Replace `pyproject.toml` dependencies with a `pixi.toml`** at the repo root:

```toml
[workspace]
channels = ["conda-forge", "pytorch", "nvidia"]
platforms = ["linux-64"]

[dependencies]
python = "3.10.*"

[pypi-dependencies]
# shared inference packages
lxml = "*"
click = "*"
rich = "*"
pyyaml = "*"
pandas = ">=2.0.0,<2.3.0"
tabulate = ">=0.8.10,<1.0.0"
jiwer = ">=3.0.0"
wandb = ">=0.25.0"

[feature.main.dependencies]
pytorch = "2.4.*"          # resolved jointly with torchvision by conda
torchvision = "0.19.*"

[feature.main.pypi-dependencies]
yaltai = {version = "*", no-deps = true}
kraken = ">=5.3.0,<6.0.0"
ultralytics = ">=8.4.8"
opencv-python = ">=4.8.0"
mean-average-precision = ">=2024.1.5.0"

[feature.train.dependencies]
pytorch = ">=2.5.0"        # required by unsloth 2025.x
torchvision = "*"

[feature.train.pypi-dependencies]
unsloth = ">=2025.1"
trl = "*"
peft = "*"
accelerate = "*"
bitsandbytes = "*"
transformers = ">=4.46.0"
datasets = ">=2.14.0"
qwen-vl-utils = "*"

[feature.dev.pypi-dependencies]
pytest = ">=8.0.0"
pytest-cov = ">=6.0.0"
pytest-mock = ">=3.14.0"
black = ">=24.0.0"
isort = ">=5.13.0"
ruff = ">=0.4.0"

[environments]
main = ["main", "dev"]
train = ["train"]
```

2. **Run `pixi install`** to generate `pixi.lock`, then commit the lock file.

3. **Replace `scripts/setup_envs.sh`** with two commands:

```bash
pixi install -e main
pixi install -e train
```

4. **Activate environments** with:
```bash
pixi shell -e main      # was: source envs/main/bin/activate
pixi shell -e train     # was: source envs/vlm-training/bin/activate
```

### Minimal intermediate fix (if pixi migration is not immediately feasible)

Generate locked `requirements.txt` files from the currently working environments and commit
them. Then rebuild from the lock:

```bash
# Snapshot the working vlm-training env
source envs/vlm-training/bin/activate
pip freeze > requirements/vlm-training.lock.txt
deactivate

# Rebuild from lock (reproducible)
virtualenv -p python3.10 envs/vlm-training-new
source envs/vlm-training-new/bin/activate
pip install -r requirements/vlm-training.lock.txt
deactivate
```

Do the same for `main`. This does not fix the structural problem but makes the current state
reproducible while the pixi migration is underway.

---

## Bug Fixed Before Migration

**`src/tasks/htr/base_vlm_htr.py`** — `load()` raised `RuntimeError` if unsloth was absent,
blocking all VLM inference in the `main` environment. The check was also redundant: the only
actual use of `FastVisionModel` was inside an `elif False:` dead-code branch. Fix applied:
removed the unsloth guard from `load()` and deleted the dead branch.

**Consequence for `main` env**: VLM inference (VLMLineHTR, VLMPageHTR, VLMMultiLineHTR) now
works without unsloth, provided `peft` is installed (needed for LoRA adapter loading). Add
`peft` to the `[feature.main]` dependencies in the future `pixi.toml`, or to pyproject.toml
`dependencies` if staying on virtualenvs.

---

## How to Verify the Migration Did Not Break Anything

### Layer 1 — Import smoke tests (run first, fast)

```bash
# main environment
pixi run -e main python -c "
import torch, torchvision, ultralytics, kraken, yaltai
from transformers import AutoProcessor
from peft import PeftModel
import jiwer, wandb, pandas, lxml, click, rich, matplotlib
print('main env imports OK')
print(f'  torch {torch.__version__}, torchvision {torchvision.__version__}')
"

# training environment
pixi run -e train python -c "
import torch, unsloth, peft, trl, accelerate, bitsandbytes
from datasets import load_dataset
print('train env imports OK')
print(f'  torch {torch.__version__}, unsloth {unsloth.__version__}')
"
```

### Layer 2 — Existing unit tests

The test suite in `tests/` runs entirely on CPU and does not require model weights.
It covers: ALTO XML parsing, CLI config loading, base task logic, KrakenHTR/Line, YoloLayout.

```bash
# Run against main env (all tests should pass)
pixi run -e main pytest tests/ -v

# Run against train env (yaltai-dependent tests may be skipped if not installed there)
pixi run -e train pytest tests/ -v -k "not test_yaltai_imports"
```

Add `pytest.ini` markers to tag env-specific tests:

```ini
[pytest]
markers =
    requires_main: needs kraken, yaltai, ultralytics
    requires_training: needs unsloth, peft, trl, datasets
```

Then tag `test_yaltai_imports` and `test_kraken_imports` with `@pytest.mark.requires_main`,
and add a `tests/test_imports_train.py` for the training env:

```python
# tests/test_imports_train.py
import pytest

@pytest.mark.requires_training
def test_unsloth_import():
    from unsloth import FastVisionModel
    assert FastVisionModel is not None

@pytest.mark.requires_training
def test_training_stack():
    import trl, peft, accelerate, bitsandbytes, datasets
```

### Layer 3 — CLI entry point

```bash
pixi run -e main docworkflow --help
pixi run -e main docworkflow predict --help
pixi run -e main docworkflow score --help
```

### Layer 4 — Synthetic end-to-end (no GPU, no model weights)

The existing `tests/conftest.py` already creates a synthetic image + ALTO XML. Extend it
into a minimal CLI integration test:

```bash
# Generate a tiny white-image dataset in /tmp/e2e_test/
# Run predict with a fake/stub model config → expect graceful failure at model load,
# not at import or CLI parsing level.
pixi run -e main docworkflow -c configs/example_config.yml predict -t layout -d test -o /tmp/e2e_out || true
```

### Layer 5 — Version compatibility assertions

Add a `scripts/check_versions.py` to run in CI:

```python
import torch, torchvision
tv, t = torchvision.__version__, torch.__version__

# torchvision 0.19.x requires torch 2.4.x; 0.20.x requires 2.5.x, etc.
tv_major, tv_minor = int(tv.split(".")[0]), int(tv.split(".")[1])
t_major, t_minor = int(t.split(".")[0]), int(t.split(".")[1])
expected_torch_minor = tv_minor - 15  # torchvision N.M ↔ torch 2.(M-15)

assert t_major == 2, f"Unexpected torch major: {t}"
assert t_minor == expected_torch_minor, (
    f"torchvision {tv} requires torch 2.{expected_torch_minor}.x, got torch {t}"
)
print(f"torch {t} + torchvision {tv}: compatible ✓")
```

### Summary checklist

| Check | Command | Env |
|---|---|---|
| Import smoke test | `python -c "import torch, ..."` | main + train |
| Unit tests | `pytest tests/ -v` | main |
| Training imports | `pytest tests/test_imports_train.py` | train |
| CLI help | `docworkflow --help` | main |
| Version pairing | `python scripts/check_versions.py` | main + train |

---

## Files to Update

| File | Action |
|---|---|
| `pyproject.toml` | Add `peft` to main dependencies; remove constraints that contradict vlm-training; or migrate fully to pixi |
| `scripts/setup_envs.sh` | Replace with `pixi install` calls or pip-lock-based install |
| `pixi.toml` (new) | Central manifest with feature groups per environment |
| `pixi.lock` (new, generated) | Commit to repo; this is the reproducibility guarantee |
| `src/tasks/htr/base_vlm_htr.py` | **Done** — removed unsloth guard and dead `elif False:` branch from `load()` |
