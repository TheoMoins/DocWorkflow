"""Import smoke tests for the train environment (unsloth, peft, trl, datasets).

These tests are tagged requires_training and are meant to run in the train env.
In the main env (no GPU, no unsloth), they skip automatically.
"""
import pytest


def _require_unsloth():
    """Skip the test if unsloth cannot initialise (missing GPU or not installed)."""
    try:
        import unsloth  # noqa: F401
    except (ImportError, NotImplementedError) as e:
        pytest.skip(f"unsloth not available: {e}")


@pytest.mark.requires_training
def test_unsloth_import():
    _require_unsloth()
    from unsloth import FastVisionModel
    assert FastVisionModel is not None


@pytest.mark.requires_training
def test_training_stack():
    _require_unsloth()
    import trl, peft, accelerate, bitsandbytes, datasets  # noqa: F401


@pytest.mark.requires_training
def test_alto_lines_in_train_env():
    """ALTO lxml parser must work in the train env (no kraken required)."""
    from src.alto.alto_lines import read_lines_geometry
    assert callable(read_lines_geometry)
