"""Tests pour la tâche de segmentation de ligne avec YOLO."""
import pytest
from unittest.mock import MagicMock, patch

from src.tasks.line.yolo_line import YoloLineTask


@pytest.fixture
def yolo_config():
    """Configuration minimale pour un entraînement de ligne."""
    return {
        "pretrained_w": "yolo26s.pt",
        "device": "cpu",
        "use_wandb": False,
        "img_size": 640,
        "batch_size": 16,
        "epochs": 50,
    }


@pytest.fixture
def trained_task(yolo_config):
    """YoloLineTask dont le modèle est un mock déjà chargé."""
    def _make(**overrides):
        with patch('src.tasks.line.yolo_line.YOLO') as mock_yolo, \
             patch('src.tasks.line.yolo_line.os.path.exists', return_value=True):
            mock_yolo.return_value = MagicMock()
            task = YoloLineTask({**yolo_config, **overrides})
            task.load("pretrained")
        return task
    return _make


def test_train_without_augmentations_keeps_ultralytics_defaults(trained_task):
    """Non-régression : une config antérieure à l'ajout de fliplr/mosaic doit
    s'entraîner exactement comme avant."""
    task = trained_task()
    task.train(data_path="dataset.yaml")

    _, kwargs = task.model.train.call_args
    assert kwargs["fliplr"] == 0.5
    assert kwargs["mosaic"] == 1.0
    assert kwargs["name"] == "Line_Segmentation_YOLO_640px_16bs_50e"
    assert kwargs["project"] == "LS-training"


def test_train_forwards_augmentations(trained_task):
    """Le chemin complet : config → train() → model.train()."""
    task = trained_task(img_size=1280, batch_size=8, epochs=30,
                        fliplr=0.0, mosaic=0.2,
                        train_name="line_det_yolo26s_1280px_8bs_30e_fl0.0_mo0.2")
    task.train(data_path="dataset.yaml", seed=7)

    _, kwargs = task.model.train.call_args
    assert kwargs["fliplr"] == 0.0
    assert kwargs["mosaic"] == 0.2
    assert kwargs["imgsz"] == 1280
    assert kwargs["epochs"] == 30
    assert kwargs["seed"] == 7
    assert kwargs["name"] == "line_det_yolo26s_1280px_8bs_30e_fl0.0_mo0.2"


@pytest.mark.parametrize("key,value", [("fliplr", 0), ("mosaic", 0.0)])
def test_zero_is_not_confused_with_absent(trained_task, key, value):
    """0 est la valeur utile du balayage : un test de vérité un peu rapide la
    remplacerait par le défaut d'ultralytics sans rien signaler."""
    task = trained_task(**{key: value})
    task.train(data_path="dataset.yaml")

    _, kwargs = task.model.train.call_args
    assert kwargs[key] == value


def test_train_args_passes_arbitrary_kwargs(trained_task):
    task = trained_task(train_args={"scale": 0.2, "patience": 20})
    task.train(data_path="dataset.yaml")

    _, kwargs = task.model.train.call_args
    assert kwargs["scale"] == 0.2
    assert kwargs["patience"] == 20
    assert kwargs["fliplr"] == 0.5  # non mentionné : reste au défaut


def test_root_key_wins_over_train_args(trained_task):
    task = trained_task(fliplr=0.0, train_args={"fliplr": 0.5})
    task.train(data_path="dataset.yaml")

    assert task.model.train.call_args[1]["fliplr"] == 0.0


def test_train_args_cannot_shadow_a_derived_key(trained_task):
    """imgsz vient de `img_size` et sert aussi à nommer le dossier de run : le
    redéfinir dans train_args doit casser bruyamment, pas produire un run dont
    le nom ment sur les hyperparamètres."""
    task = trained_task(train_args={"imgsz": 1536})
    with pytest.raises(TypeError, match="imgsz"):
        task.train(data_path="dataset.yaml")


def test_train_without_data_raises(trained_task):
    task = trained_task()
    with pytest.raises(ValueError, match="No training data"):
        task.train()
