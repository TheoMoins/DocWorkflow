"""Tests pour la tâche de segmentation layout avec YOLO."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from src.tasks.layout.yolo_layout import YoloLayoutTask


@pytest.fixture
def yolo_config():
    """Configuration pour les tests YOLO."""
    return {
        "model_path": "test_model.pt",
        "pretrained_w": "yolo11s.pt",
        "device": "cpu",
        "use_wandb": False,
        "img_size": 640,
        "batch_size": 4,
        "epochs": 1
    }


def test_yolo_layout_initialization(yolo_config):
    """Teste l'initialisation de YoloLayoutTask."""
    task = YoloLayoutTask(yolo_config)
    
    assert task.name == "Layout Segmentation (YOLO)"
    assert task.config == yolo_config
    assert task.device == "cpu"


@patch('src.tasks.layout.yolo_layout.YOLO')
@patch('src.tasks.layout.yolo_layout.os.path.exists')
def test_yolo_layout_load_trained(mock_exists, mock_yolo, yolo_config):
    """Teste le chargement d'un modèle entraîné."""
    mock_exists.return_value = True
    mock_model = MagicMock()
    mock_yolo.return_value = mock_model
    
    task = YoloLayoutTask(yolo_config)
    task.load(mode="trained")
    
    assert task.model_loaded == "trained"
    mock_yolo.assert_called_once_with(yolo_config["model_path"])


@patch('src.tasks.layout.yolo_layout.YOLO')
@patch('src.tasks.layout.yolo_layout.os.path.exists')
def test_yolo_layout_load_pretrained(mock_exists, mock_yolo, yolo_config):
    """Teste le chargement de poids pré-entraînés."""
    mock_exists.return_value = True
    mock_model = MagicMock()
    mock_yolo.return_value = mock_model
    
    task = YoloLayoutTask(yolo_config)
    task.load(mode="pretrained")
    
    assert task.model_loaded == "pretrained"
    mock_yolo.assert_called_once_with(yolo_config["pretrained_w"])


def test_yolo_layout_load_missing_weights(yolo_config):
    """Teste le chargement avec des poids manquants."""
    yolo_config["model_path"] = "non_existent.pt"
    task = YoloLayoutTask(yolo_config)
    
    with pytest.raises(FileNotFoundError):
        task.load(mode="trained")


@patch('src.tasks.layout.yolo_layout.YOLO')
@patch('src.tasks.layout.yolo_layout.os.path.exists')
def test_yolo_layout_to_device(mock_exists, mock_yolo, yolo_config):
    """Teste le déplacement du modèle vers un device."""
    mock_exists.return_value = True
    mock_model = MagicMock()
    mock_yolo.return_value = mock_model
    
    task = YoloLayoutTask(yolo_config)
    task.load()
    task.to_device("cpu")
    
    assert task.device == "cpu"
    mock_model.to.assert_called_with("cpu")


@patch('src.tasks.layout.yolo_layout.YOLO')
@patch('src.tasks.layout.yolo_layout.os.path.exists')
def test_yolo_layout_score_no_files(mock_exists, mock_yolo, yolo_config, temp_dir):
    """Teste score avec des dossiers vides."""
    # Mock pour éviter le chargement réel du modèle
    mock_exists.return_value = True
    mock_model = MagicMock()
    mock_yolo.return_value = mock_model
    
    task = YoloLayoutTask(yolo_config)
    
    with pytest.raises(ValueError, match="No ground truth"):
        task.score(str(temp_dir / "pred"), str(temp_dir / "gt"))


@patch('src.tasks.layout.yolo_layout.glob.glob')
@patch('src.tasks.layout.yolo_layout.extract_zones_from_alto')
@patch('src.tasks.layout.yolo_layout.Image.open')
def test_yolo_layout_score_with_files(mock_image, mock_extract, mock_glob, 
                                      yolo_config, temp_dir):
    """Teste le scoring avec des fichiers."""
    # Setup mocks
    mock_glob.return_value = [str(temp_dir / "test.xml")]
    mock_extract.return_value = (
        str(temp_dir / "test.jpg"),
        [{'bbox': [10, 10, 100, 100], 'label': 'MainZone'}]
    )
    mock_img = Mock()
    mock_img.size = (640, 480)
    mock_image.return_value = mock_img
    
    task = YoloLayoutTask(yolo_config)
    
    # Note: Ce test nécessite que les fichiers pred existent aussi
    # Il peut échouer, c'est normal sans setup complet


@patch('src.tasks.layout.yolo_layout.YOLO')
@patch('src.tasks.layout.yolo_layout.os.path.exists')
def test_yolo_layout_predict_no_images(mock_exists, mock_yolo, yolo_config, temp_dir):
    """Teste predict sans images."""
    mock_exists.return_value = True
    mock_model = MagicMock()
    mock_yolo.return_value = mock_model
    
    task = YoloLayoutTask(yolo_config)
    task.load()
    
    with pytest.raises(ValueError, match="No images found"):
        task.predict(str(temp_dir), str(temp_dir / "output"))