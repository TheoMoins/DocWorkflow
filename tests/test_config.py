"""Tests pour le système de configuration."""
import pytest
from src.cli.config import Config
from src.cli.config.exceptions import InvalidConfigValue


def test_config_loading(sample_config_file):
    """Teste le chargement d'un fichier de configuration."""
    config = Config(str(sample_config_file))
    assert config.yaml is not None
    assert config.yaml["run_name"] == "test_run"


def test_config_tasks_property(sample_config_file):
    """Teste la propriété tasks."""
    config = Config(str(sample_config_file))
    tasks = config.get_tasks()
    assert "layout" in tasks


def test_config_data_property(sample_config_file):
    """Teste la propriété data."""
    config = Config(str(sample_config_file))
    data = config.data
    assert "test" in data


def test_config_invalid_task_type(temp_dir):
    """Teste la gestion d'un type de tâche invalide."""
    import yaml
    invalid_config = {
        "device": "cpu",
        "use_wandb": False,
        "tasks": {
            "layout": {
                "type": "InvalidTask",
                "config": {
                    "model_path": "test.pt"
                }
            }
        }
    }
    config_path = temp_dir / "invalid_config.yml"
    with open(config_path, 'w') as f:
        yaml.dump(invalid_config, f)
    
    config = Config(str(config_path))
    
    with pytest.raises(InvalidConfigValue):
        _ = config.layout_task

        
def test_config_add_global_params(sample_config_file):
    """Teste l'ajout des paramètres globaux."""
    config = Config(str(sample_config_file))
    task_config = {}
    updated = config.add_global_params_to_config(task_config)
    assert "device" in updated
    assert "use_wandb" in updated