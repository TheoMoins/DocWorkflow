from core.base_model import BaseModel
import torch
import os
import glob
from ultralytics import YOLO, settings
from datetime import datetime
from pathlib import Path

from core.yolalto import (
    parse_yolo_results, remove_duplicates, 
    create_alto_xml, save_alto_xml, bbox_baseline
)

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
        settings.update({"wandb": config.get('use_wandb', True)})

        self.wandb_project = "LA-comparison"
        self.model_loaded = None

    def load(self, mode="trained"):
        """
        Load the model from a weights file.
        
        Args:
            mode: "trained" or "pretrained", indicate if a trained model (for prediction/evaluation)
            or pretrained weigths (for training) are given
        """

        if mode == "pretrained":
            if not os.path.exists(self.config["pretrained_w"]):
                raise FileNotFoundError(f"Pretrained weights file not found: {self.config['pretrained_w']}")
            else:
                model = self.config["pretrained_w"]
                self.model_loaded = "pretrained"

        elif mode == "trained":
            if not os.path.exists(self.config["model_path"]):
                raise FileNotFoundError(f"Trained weights file not found: {self.config['model_path']}")
            else:
                model = self.config["model_path"]
                self.model_loaded = "trained"

        if "docyolo" in self.config['name']:
            from doclayout_yolo import YOLOv10
            self.model = YOLOv10(model)
        else:
            self.model = YOLO(model)
        
        self.to_device()
    
    def train(self, data_path=None):
        """
        Train the layout model.
        
        Args:
            data_path: Path to training data
            **kwargs: Additional training arguments
        """

        if self.model_loaded != "pretrained":
            self.load("pretrained")

        training_data = data_path
        if not data_path:
            if not self.config["data_path"]:
                raise ValueError("No training data is provided in the config file or in the argument of the train function.")
            else:
                training_data = self.config["data_path"]

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
        if self.model_loaded != "trained":
           self.load("trained")
        
        # Evaluate on the test set
        metrics_set = self.model.val(data=path, split='test')
        
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
    

    def predict(self, output_dir):
        """
        Perform prediction on one image or on all images in corpus_path.
        
        Args:
            image_path: Optional path to a single image. If None, use corpus_path from config.
                
        Returns:
            Prediction results for single image or a dictionary of results for multiple images
        """
        if self.model_loaded != "trained":
           self.load("trained")

        # Otherwise, predict on all images in corpus_path
        corpus_path = self.config.get("corpus_path")
        if not corpus_path:
            raise ValueError("No corpus_path specified in config and no image_path provided")
        
        # Find all images in the corpus directory
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(corpus_path, ext)))
        
        if not image_paths:
            raise ValueError(f"No images found in {corpus_path}")
        # Process images in batches
        batch_size = 4  # Default batch size
        num_batches = len(image_paths) // batch_size + (1 if len(image_paths) % batch_size else 0)
        
        print(f"Processing {len(image_paths)} images in {num_batches} batches...")
        
        for batch_idx in range(num_batches):
            batch = image_paths[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            
            try:
                # Run model on batch
                results = self.model(batch)
                
                # Parse results
                batch_detections = parse_yolo_results(results)
                # Process each image result
                for image_path, (detections, wh) in zip(batch, batch_detections):
                    output_path = os.path.join(output_dir, Path(image_path).with_suffix('.xml').name)
                    # Remove duplicate detections
                    detections = remove_duplicates(detections)
                    
                    # Create ALTO XML and save it
                    alto = create_alto_xml(detections, image_path, wh)
                    save_alto_xml(alto, output_path)
                                
            except Exception as e:
                print(f"Error processing batch: {e}")
                import traceback
                traceback.print_exc()

        return results