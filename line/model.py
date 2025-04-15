from core.base_model import BaseModel
import torch
import os
from tqdm import tqdm
from PIL import Image
import glob
import numpy as np
from mean_average_precision import MetricBuilder
from datetime import datetime
import gc

import threading

from kraken.lib.vgsl import TorchVGSLModel
from kraken.kraken import SEGMENTATION_DEFAULT_MODEL
from kraken.lib.xml import XMLPage
from yaltai.models.krakn import segment as line_segment

from core.utils import extract_lines_from_alto, convert_lines_to_boxes

class LineModel(BaseModel):
    """
    Class for line segmentation models.
    """
    
    def __init__(self, config):
        """
        Initialize the line segmentation model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)

        self.wandb_project = "LS-comparison"
    
    def load(self):
        """
        Load the model from a weights file.
        """
        model_path = self.config.get('model_path', SEGMENTATION_DEFAULT_MODEL)
        self.model = TorchVGSLModel.load_model(model_path)
        self.to_device()
    
    def train(self, data_path, **kwargs):
        """
        Train the line segmentation model.
        
        Args:
            data_path: Path to training data
            **kwargs: Additional training arguments
        """
        # TODO : Line segmentation model training is not yet implemented
        print("Training for line segmentation models is not yet implemented.")
        

    def _predict_page(self, image, file_name):
        """
        Get line predictions for an image already segmented at the layout scale.
        
        Args:
            image: Image object
            file_name: Path to ALTO file containing layout segmentation
            
        Returns:
            Numpy array of prediction boxes
        """
        # Utilisez une liste pour stocker le résultat du thread principal
        result = [None]
        # Flag pour indiquer si un timeout s'est produit
        is_timeout = [False]
        
        def process_image():
            try:
                # Extract regions from ALTO file
                _, _, alto_regions = extract_lines_from_alto(file_name)
                
                if not alto_regions:
                    raise ValueError(f"No regions available in ALTO file {file_name}")
                
                # Perform line segmentation
                segmentation_result = line_segment(
                    image,
                    text_direction=self.config["text_direction"],
                    model=self.model,
                    device=self.device, 
                    regions=alto_regions,
                    ignore_lignes=False
                )
                
                # Extract predicted lines
                lines = [{'baseline': line.baseline, 'boundary': line.boundary, 'id': line.id}
                        for line in segmentation_result.lines]
                
                # Stocker le résultat seulement si aucun timeout ne s'est produit
                if not is_timeout[0]:
                    result[0] = lines
            except Exception as e:
                print(f"Error during line segmentation: {e}")
                if not is_timeout[0]:
                    result[0] = []
        
        # Créer et démarrer le thread de traitement
        thread = threading.Thread(target=process_image)
        thread.daemon = True
        thread.start()
        
        # Attendre que le thread se termine, avec un timeout
        thread.join(timeout=180)
        
        # Si le thread est toujours en vie après le timeout
        if thread.is_alive():
            is_timeout[0] = True
            print(f"\nTimeout for {file_name}: Interruption after 180 seconds")
            # Le thread continuera à s'exécuter en arrière-plan, mais nous ignorons son résultat
            return []
        
        # Retourner le résultat (vide en cas d'erreur)
        return result[0] if result[0] is not None else []

    def _compute_metrics(self, test_path, corpus_path=None):
        """
        Compute metrics for the line segmentation model.
        
        Args:
            test_path: Path to test data
            corpus_path: Optional path to corpus data
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.model:
            self.load()
        
        # Initialize metrics builder
        builder = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=1)
        
        # Process ground truth data
        gt_files = sorted(glob.glob(os.path.join(test_path, "*.xml")))
        if not gt_files:
            raise ValueError(f"No ground truth ALTO files found in {test_path}")
        
        # Process files
        metrics_by_image = []
        for gt_file in tqdm(gt_files, desc="Processing pages", unit="page"):
            image_path, gt_lines, _ = extract_lines_from_alto(gt_file)
            if not gt_lines:
                print(f"Warning: No lines found in {gt_file}")
                continue
            
            try:
                image = Image.open(image_path)
                image_size = image.size
            except Exception as e:
                print(f"Error opening image {image_path}: {e}")
                continue
            
            pred_lines = self._predict_page(image, gt_file)
            
            if pred_lines:
                # Convert ground truth and predictions to the format expected by MAP
                gt_boxes = convert_lines_to_boxes(gt_lines, image_size, is_gt=True)
                pred_boxes = convert_lines_to_boxes(pred_lines, image_size, is_gt=False)
                
                if gt_boxes.shape[0] > 0 and pred_boxes.shape[0] > 0:
                    builder.add(pred_boxes, gt_boxes)
                    
            if image:
                    image.close()
                    del image
                
            gc.collect()
        
        # Calculate metrics
        metrics = builder.value(iou_thresholds=[round(x, 2) for x in np.arange(0.5, 1.0, 0.05)])
        # Extract metrics
        # TODO : ADD mAP/precision/recall for MainZone
        test_metrics = {
            "dataset_test/map50-95": metrics["mAP"],
            "dataset_test/map50": metrics[0.5][0]["ap"],
            "dataset_test/map75": metrics[0.75][0]["ap"],
            "dataset_test/precision": metrics[0.75][0]["precision"].mean(),
            "dataset_test/recall": metrics[0.75][0]["recall"].mean()
        }
        
        # TODO : Process corpus data if provided (similar logic to test data)
        if corpus_path:
            corpus_metrics = {
                # Similar to test metrics, but for corpus data
                # Implementation would be similar to the test metrics logic
            }
            combined_metrics = {**test_metrics, **corpus_metrics}
        else:
            combined_metrics = test_metrics
    
        return combined_metrics

    
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
            
        # Load the image
        image = Image.open(image_path)
                
        # Get regions if a layout model is specified
        regions = None
        if "layout_model_path" in self.config and self.config["layout_model_path"]:
            # Get regions from the layout model
            layout_model_path = self.config["layout_model_path"]
            # TODO : Change this part !
            if "docyolo" in layout_model_path:
                try:
                    from doclayout_yolo import YOLOv10
                    layout_model = YOLOv10(layout_model_path)
                except ImportError:
                    from ultralytics import YOLO
                    layout_model = YOLO(layout_model_path)
            else:
                from ultralytics import YOLO
                layout_model = YOLO(layout_model_path)
            
            layout_model.to(self.device)
            from yaltai.models.yolo import segment as zone_segment
            regions = zone_segment(layout_model, image_path)
        
        # Perform line segmentation
        segmentation_result = line_segment(
            image,
            text_direction=self.config.get("text_direction", "horizontal-lr"),
            model=self.model,
            device=self.device, 
            regions=regions,
            ignore_lignes=False
        )
        
        return segmentation_result