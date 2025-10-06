"""Tests pour la classe BaseTask."""
import pytest
from src.tasks.base_tasks import BaseTask


class DummyTask(BaseTask):
    """Tâche factice pour les tests."""
    
    def load(self):
        self.model = "dummy_model"
    
    def train(self, **kwargs):
        pass
    
    def predict(self, data_path, output_dir, save_image=False):
        return []
    
    def score(self, pred_path, gt_path):
        return {"metric": 0.5}


def test_base_task_initialization():
    """Teste l'initialisation d'une tâche."""
    config = {"device": "cpu", "use_wandb": False}
    task = DummyTask(config)
    
    assert task.config == config
    assert task.device == "cpu"
    assert task.use_wandb is False


def test_base_task_to_device():
    """Teste le déplacement vers un device."""
    config = {"device": "cpu"}
    task = DummyTask(config)
    task.model = None
    
    task.to_device("cpu")
    assert task.device == "cpu"


def test_base_task_display_metrics():
    """Teste l'affichage des métriques."""
    config = {"device": "cpu"}
    task = DummyTask(config)
    
    metrics = {
        "test/map": 0.85,
        "test/precision": 0.90
    }
    
    # Should not raise
    task._display_metrics(metrics)