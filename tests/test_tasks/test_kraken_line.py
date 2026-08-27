"""Tests pour la tâche de segmentation de lignes avec Kraken."""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.tasks.line.kraken_line import KrakenLineTask


@pytest.fixture
def kraken_line_config():
    """Configuration pour les tests Kraken Line."""
    return {
        "model_path": "baseline_model.mlmodel",
        "text_direction": "horizontal-lr",
        "device": "cpu",
        "use_wandb": False
    }


def test_kraken_line_initialization(kraken_line_config):
    """Teste l'initialisation de KrakenLineTask."""
    task = KrakenLineTask(kraken_line_config)
    
    assert task.name == "Line Segmentation (Kraken)"
    assert task.config == kraken_line_config


@patch('src.tasks.line.kraken_line.TorchVGSLModel.load_model')
def test_kraken_line_load(mock_load_model, kraken_line_config):
    """Teste le chargement du modèle."""
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    
    task = KrakenLineTask(kraken_line_config)
    task.load()
    
    mock_load_model.assert_called_once_with(kraken_line_config["model_path"])
    assert task.model is not None


def test_kraken_line_train_not_implemented(kraken_line_config):
    """Teste que train n'est pas encore implémenté."""
    task = KrakenLineTask(kraken_line_config)
    
    # Ne devrait pas crasher mais afficher un message
    task.train("dummy_path")


def test_kraken_line_score_no_files(kraken_line_config, temp_dir):
    """Teste score sans fichiers."""
    task = KrakenLineTask(kraken_line_config)

    gt_dir = temp_dir / "gt"
    gt_dir.mkdir()

    with pytest.raises(ValueError, match="No files found"):
        task.score(str(temp_dir / "pred"), str(gt_dir))


@patch('src.tasks.line.kraken_line.TorchVGSLModel.load_model')
def test_kraken_line_predict_no_images(mock_load, kraken_line_config, temp_dir):
    """Teste predict sans images."""
    mock_model = MagicMock()
    mock_load.return_value = mock_model
    
    task = KrakenLineTask(kraken_line_config)
    task.load()
    
    with pytest.raises(ValueError, match="No files found"):
        task.predict(str(temp_dir), str(temp_dir / "output"))


def test_kraken_line_text_direction_config(kraken_line_config):
    """Teste que la direction du texte est correctement configurée."""
    task = KrakenLineTask(kraken_line_config)
    
    assert task.config["text_direction"] == "horizontal-lr"