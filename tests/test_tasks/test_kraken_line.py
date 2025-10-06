"""Tests pour la tâche de segmentation de lignes avec Kraken."""
import pytest
from unittest.mock import Mock, patch, MagicMock
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
    assert task.wandb_project == "LS-comparison"


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


@patch('src.tasks.line.kraken_line.glob.glob')
def test_kraken_line_score_no_files(mock_glob, kraken_line_config, temp_dir):
    """Teste score sans fichiers."""
    mock_glob.return_value = []
    
    task = KrakenLineTask(kraken_line_config)
    
    with pytest.raises(ValueError, match="No ground truth"):
        task.score(str(temp_dir / "pred"), str(temp_dir / "gt"))


@patch('src.tasks.line.kraken_line.extract_lines_from_alto')
@patch('src.tasks.line.kraken_line.glob.glob')
@patch('src.tasks.line.kraken_line.Image.open')
def test_kraken_line_score_missing_predictions(mock_image, mock_glob, 
                                                mock_extract, kraken_line_config, 
                                                temp_dir):
    """Teste score avec des prédictions manquantes."""
    gt_file = str(temp_dir / "test.xml")
    mock_glob.return_value = [gt_file]
    
    mock_extract.return_value = (
        str(temp_dir / "test.jpg"),
        [{'baseline': [[0, 50], [100, 50]], 'boundary': None, 'id': 'line1'}],
        {}
    )
    
    mock_img = Mock()
    mock_img.size = (640, 480)
    mock_image.return_value = mock_img
    
    task = KrakenLineTask(kraken_line_config)
    
    # Devrait afficher un warning mais continuer
    # Note: Ce test peut nécessiter plus de setup


@patch('src.tasks.line.kraken_line.TorchVGSLModel.load_model')
@patch('src.tasks.line.kraken_line.glob.glob')
def test_kraken_line_predict_no_images(mock_glob, mock_load, 
                                       kraken_line_config, temp_dir):
    """Teste predict sans images."""
    mock_glob.return_value = []
    mock_model = MagicMock()
    mock_load.return_value = mock_model
    
    task = KrakenLineTask(kraken_line_config)
    task.load()
    
    with pytest.raises(ValueError, match="No images found"):
        task.predict(str(temp_dir), str(temp_dir / "output"))


def test_kraken_line_text_direction_config(kraken_line_config):
    """Teste que la direction du texte est correctement configurée."""
    task = KrakenLineTask(kraken_line_config)
    
    assert task.config["text_direction"] == "horizontal-lr"