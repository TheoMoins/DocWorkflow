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
        config["wandb_project"] = self.yaml.get("wandb_project", None)
        config["run_name"] = self.yaml.get("run_name", "unknown")
        config["data"] = self.yaml.get("data", {})
        config["reading_order"] = self.yaml.get("reading_order", "dbscan")
        return config

    def get_tasks(self) -> list:
        results = []
        for task_name in ['layout', 'line', 'htr']:
            task_obj = getattr(self, f"{task_name}_task")
            
            if task_obj is None:
                continue

            results.append(task_name)
        return results

    @staticmethod
    def _find_first_xml(directory: Path) -> Path:
        """
        Premier fichier XML du dossier, à la racine puis dans les sous-dossiers.

        Un dataset peut être plat (fichiers à la racine) ou hiérarchique
        (fichiers un niveau plus bas) ; on explore les deux, sur un seul niveau
        comme le fait `discover_dataset_structure`.
        """
        for xml_file in sorted(directory.glob("*.xml")):
            return xml_file

        for subdir in sorted(p for p in directory.iterdir()
                             if p.is_dir() and not p.name.startswith('.')):
            for xml_file in sorted(subdir.glob("*.xml")):
                return xml_file

        return None

    def get_scoreable_tasks(self, pred_path: str, gt_path: str) -> list:
        scoreable = []

        if not pred_path or not gt_path:
            return scoreable

        pred_dir, gt_dir = Path(pred_path), Path(gt_path)
        if not pred_dir.is_dir() or not gt_dir.is_dir():
            return scoreable

        pred_file = self._find_first_xml(pred_dir)
        gt_file = self._find_first_xml(gt_dir)

        if pred_file is None or gt_file is None:
            return scoreable

        tree = ET.parse(str(pred_file))
        root = tree.getroot()
        ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}

        # Vérifier chaque tâche configurée
        for task_name in ['layout', 'line', 'htr']:
            task_obj = getattr(self, f"{task_name}_task")

            if task_obj is None:
                continue

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
            value = self.yaml["data"][set_type]
            if isinstance(value, list):
                result[set_type] = [Path(p) for p in value]
            else:
                result[set_type] = Path(value)
        return result
    
    @classmethod
    def import_class(cls, name: str) -> object:
        try:
            module_name, _class = ModelImports[name.upper()].value
        except KeyError:
            raise InvalidConfigValue(name)
        module = import_module(module_name)
        return getattr(module, _class)

    @classmethod
    def create_class(cls, code_name: str, params: dict) -> object:
        _class = cls.import_class(name=code_name)
        return _class(params)