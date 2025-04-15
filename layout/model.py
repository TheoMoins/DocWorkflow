from core.base_model import BaseModel
import torch
import os
import wandb
from ultralytics import YOLO, settings
from datetime import datetime

class LayoutModel(BaseModel):
    """
    Class for layout segmentation models.
    """
    
    def __init__(self, config):
        """
        Initialize the layout model.
        
        Args:
            config: Model configuration dictionary
            models_dir: Directory containing model weights
        """
        super().__init__(config)
        settings.update({"wandb": True})

        self.wandb_project = "LA-comparison"
    
    def load(self):
        """
        Load the model from a weights file.
        """
        if "docyolo" in self.config['name']:
            from doclayout_yolo import YOLOv10
            self.model = YOLOv10(self.config["model_path"])
        else:
            self.model = YOLO(self.config["model_path"])
        
        self.to_device()
    
    def train(self, data_path=None):
        """
        Train the layout model.
        
        Args:
            data_path: Path to training data
            **kwargs: Additional training arguments
        """

        if not self.model:
            self.load()

        training_data = data_path
        if not data_path:
            if not self.data_path:
                raise ValueError("No training data is provided in the config file or in the argument of the train function.")
            else:
                training_data = self.data_path

        # Train the model
        self.model.train(
            data=training_data, 
            project='LA-training', 
            imgsz=self.config["img_size"],
            batch=self.config["batch_size"],
            epochs=self.config["epochs"], 
            name=self.config["name"]
        )
    
    def _compute_metrics(self, path, is_corpus=False):
        """
        Compute metrics for the layout model.
        
        Args:
            path: Path to test data
            is_corpus: Indicate if it's corresponding to an additional test or not
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.model:
            self.load()
        
        # Evaluate on the test set
        metrics_set = self.model.val(data=path, split='test')
        val_dirs = [d for d in os.listdir('runs/detect') if d.startswith('val')]
        last_val_dir = sorted(val_dirs)[-1]  # Take the last val directory
        test_run_path = os.path.join('runs/detect', last_val_dir)
        
        if is_corpus:
            metrics = {
                "dataset_test/map50-95": metrics_set.box.map,
                "dataset_test/map50": metrics_set.box.map50,
                "dataset_test/map75": metrics_set.box.map75,
                "dataset_test/precision": metrics_set.box.mp,
                "dataset_test/recall": metrics_set.box.mr,
                "MainZone_test_set/map50-95_MainZone": metrics_set.box.maps[1], 
                "MainZone_test_set/precision_MainZone": metrics_set.box.p[1],    
                "MainZone_test_set/recall_MainZone": metrics_set.box.r[1]   
            }
        
        else:
            metrics = {
                "dataset_corpus/map50-95": metrics_set.box.map,
                "dataset_corpus/map50": metrics_set.box.map50,
                "dataset_corpus/map75": metrics_set.box.map75,
                "dataset_corpus/precision": metrics_set.box.mp,
                "dataset_corpus/recall": metrics_set.box.mr,
                "MainZone_corpus/map50-95_MainZone": metrics_set.box.maps[1], 
                "MainZone_corpus/precision_MainZone": metrics_set.box.p[1],    
                "MainZone_corpus/recall_MainZone": metrics_set.box.r[1]   
            }
                    
        return metrics
    
    def predict(self, image_path):
        """
        Perform prediction on an image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Prediction results
        """
        if not self.model:
            self.load()
            
        return self.model(image_path)