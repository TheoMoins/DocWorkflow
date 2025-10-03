from importlib import import_module

from yaml import safe_load
from pathlib import Path
from lxml import etree as ET

from .constants import ModelImports
from .exceptions import InvalidConfigValue

from src.tasks.base_tasks import BaseTask


class Config(object):
    def __init__(self, config_path: str) -> None:
        with open(config_path, "r") as f:
            self.yaml: dict = safe_load(f)

        self._validate_parameter_consistency()

    def _validate_parameter_consistency(self) -> None:
        pass

    def add_global_params_to_config(self, config):
        config["device"] = self.yaml.get("device", "cpu")
        config["use_wandb"] = self.yaml.get("use_wandb", False)
        config["wandbproject"] = self.yaml.get("wandbproject", None)
        return config

    def get_tasks(self) -> list:
        results = []
        for task_name in ['layout', 'line', 'htr']:
            task_obj = getattr(self, f"{task_name}_task")
            
            if task_obj is None:
                continue

            results.append(task_name)
        return results

    def get_scoreable_tasks(self, pred_path: str, gt_path: str) -> list:        
        scoreable = []
        pred_files = list(Path(pred_path).glob("*.xml"))
        gt_files = list(Path(gt_path).glob("*.xml"))
        
        if not pred_files or not gt_files:
            return scoreable
        
        # Vérifier chaque tâche configurée
        for task_name in ['layout', 'line', 'htr']:
            task_obj = getattr(self, f"{task_name}_task")
            
            if task_obj is None:
                continue

            tree = ET.parse(str(pred_files[0]))
            root = tree.getroot()
            ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
            
            if task_name == 'layout':
                # Vérifier qu'il y a des TextBlocks
                if len(root.findall('.//alto:TextBlock', ns)) > 0:
                    scoreable.append(task_name)
            
            elif task_name == 'line':
                # Vérifier qu'il y a des TextLines
                if len(root.findall('.//alto:TextLine', ns)) > 0:
                    scoreable.append(task_name)
            
            elif task_name == 'htr':
                # Vérifier qu'il y a du texte transcrit
                strings = root.findall('.//alto:String', ns)
                if len(strings) > 0 and any(s.get('CONTENT') for s in strings):
                    scoreable.append(task_name)
        return scoreable

    @property
    def layout_task(self) -> BaseTask:
        if not self.yaml["tasks"].get("layout"):
            return None
        name = self.yaml["tasks"]["layout"]["type"]
        config = self.yaml["tasks"]["layout"]["config"]
        config = self.add_global_params_to_config(config)
        return self.create_class(code_name=name, params=config)
    
    @property
    def line_task(self) -> BaseTask:
        if not self.yaml["tasks"].get("line"):
            return None
        name = self.yaml["tasks"]["line"]["type"]
        config = self.yaml["tasks"]["line"]["config"]
        config = self.add_global_params_to_config(config)
        return self.create_class(code_name=name, params=config)
    
    @property
    def htr_task(self) -> BaseTask:
        if not self.yaml["tasks"].get("htr"):
            return None
        name = self.yaml["tasks"]["htr"]["type"]
        config = self.yaml["tasks"]["htr"]["config"]
        config = self.add_global_params_to_config(config)
        return self.create_class(code_name=name, params=config)


    @property
    def data(self) -> dict:
        result = {}
        if not self.yaml.get("data"):
            return result
        for set_type in ["train", "valid", "test"]:
            if not self.yaml["data"].get(set_type):
                continue
            result[set_type] = Path(self.yaml["data"][set_type])
        return result
    
    @classmethod
    def import_class(cls, name: str) -> object:
        try:
            module_name, _class = ModelImports[name.upper()].value
        except ValueError:
            raise InvalidConfigValue(name)
        module = import_module(module_name)
        return getattr(module, _class)

    @classmethod
    def create_class(cls, code_name: str, params: dict) -> object:
        _class = cls.import_class(name=code_name)
        return _class(params)