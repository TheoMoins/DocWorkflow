import json
import os
from pathlib import Path

class ConfigManager:
    """
    Configuration file manager for models.
    """
    
    def __init__(self, config_dir=None):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Optional directory containing configuration files
        """
        self.config_dir = config_dir
        self.configs = {}
        
    def load_configs(self, config_path):
        """
        Load a single or multiple configuration files.
        
        Args:
            config_path: Path to the JSON configuration file
            
        Returns:
            Configuration dictionary
        """
        if isinstance(config_path, str):
            config_path_list = [config_path]
        else:
            config_path_list = config_path
        
        config_list = []
        for path in config_path_list:

            with open(path) as f:
                config = json.load(f)
            
            self.configs[config.get('name', os.path.basename(path))] = config
            config_list.append(config)

        return config_list
    
    def get_default_config(self, model_type):
        """
        Get the default configuration for a model type.
        
        Args:
            model_type: Model type ('layout' or 'line')
            
        Returns:
            Default configuration dictionary
        """
        defaults = {
            'layout': {
                'name': 'catmus_seg_s_16_10',
                'model_path': "layout/LA-training/catmus_seg_s_16_10/weights/best.pt",
                'data_path': "../data_yolo/medieval-segmentation-yolo/config.yaml",
                'corpus_path': False,
                'data': 'catmus_seg',
                'training_mode': 'restricted',
                'batch_size': 16,
                'epochs': 10,
                'pretrained_w': 'yolo11s',
                'img_size': 640,
                "use_wandb": False
            },
            'line': {
                'name': 'default_line_model',
                'data_path': "../data/minitest",
                'corpus_path': False,
                'data': 'catmus_lines',
                'training_mode': 'restricted',
                'batch_size': 16,
                'epochs': 100,
                'img_size': 1024,
                'text_direction': 'horizontal-lr',
                'iou_threshold': 0.5,
                'buffer_size': 5,
                "use_wandb": False
            }
        }
        
        return defaults.get(model_type, {})
    
    def save_config(self, config, path=None):
        """
        Save a configuration to a JSON file.
        
        Args:
            config: Configuration dictionary
            path: Optional path to save the file
            
        Returns:
            Path where the configuration was saved
        """
        if not path:
            if self.config_dir:
                path = os.path.join(self.config_dir, f"{config['name']}.json")
            else:
                path = f"{config['name']}.json"
                
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)
            
        return path