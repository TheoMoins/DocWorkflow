import os
import glob
import gc
import shutil
import warnings
import numpy as np

from ultralytics import YOLO, settings
from pathlib import Path

from mean_average_precision import MetricBuilder
from PIL import Image
from tqdm import tqdm

from src.alto.alto_zones import extract_zones_from_alto, convert_zones_to_boxes
from src.tasks.base_tasks import BaseTask


from src.alto.yolalto import (
    parse_yolo_results, remove_duplicates, 
    create_alto_xml, save_alto_xml
)

class YoloLayoutTask(BaseTask):
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

        self.name = "Layout Segmentation (YOLO)"

        self.model_loaded = None
        self.model = None

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
            if not self.config.get("model_path"):
                raise FileNotFoundError(f"Trained weights file not found: {self.config['model_path']}")
            else:
                model = self.config["model_path"]
                self.model_loaded = "trained"

        # if "DocYOLO" in self.name:
        #     from doclayout_yolo import YOLOv10
        #     self.model = YOLOv10(model)
        # else:
        self.model = YOLO(model)
        
        self.to_device()
    
    def train(self, data_path=None, seed=42):
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

        save_name = self.name + "_" + str(self.config["img_size"]) + "px_" + \
                    str(self.config["batch_size"]) + "bs_" + str(self.config["epochs"]) + "e"

        # Train the model
        self.model.train(
            data=training_data, 
            project='LA-training', 
            imgsz=self.config["img_size"],
            batch=self.config["batch_size"],
            epochs=self.config["epochs"], 
            name=save_name,
            device = self.config["device"],
            seed = seed
        )

    def score(self, pred_path, gt_path):
        """
        Compute metrics for the layout model.
        
        Args:
            pred_path: Path to directory containing prediction 
            gt_path: Path to directory containing ground truth 
            
        Returns:
            Dictionary of evaluation metrics
        """
        wandb_run = self._init_wandb()

        warnings.filterwarnings('ignore', category=FutureWarning, module='mean_average_precision')

        if self.model_loaded != "trained":
           if not self.config.get("input_file"):
               self.load("trained")
        

        # Initialize metrics builder
        builder = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=1)

        # Process ground truth data
        gt_files = sorted(glob.glob(os.path.join(gt_path, "*.xml")))
        if not gt_files:
            raise ValueError(f"No ground truth ALTO files found in {gt_path}")

        # Process files
        for gt_file in tqdm(gt_files, desc="Scoring pages", unit="page"):
            # Extract base name for matching prediction file
            base_name = os.path.basename(gt_file)
            pred_file = os.path.join(pred_path, base_name)
            
            if not os.path.exists(pred_file):
                print(f"Warning: No prediction file found for {base_name}")
                continue
            
            # Extract ground truth zones and image path
            image_path, gt_zones = extract_zones_from_alto(gt_file)
            if not gt_zones:
                print(f"Warning: No zones found in {gt_file}")
                continue
            
            # Extract predicted zones
            _, pred_zones = extract_zones_from_alto(pred_file)
            if not pred_zones:
                print(f"Warning: No predicted zones found in {pred_file}")
                continue
            
            try:
                image = Image.open(image_path)
                image_size = image.size
            except Exception as e:
                print(f"Error opening image {image_path}: {e}")
                continue
            
            # Convert ground truth and predictions to the format expected by MAP
            gt_boxes = convert_zones_to_boxes(gt_zones, image_size, is_gt=True)
            pred_boxes = convert_zones_to_boxes(pred_zones, image_size, is_gt=False)
            
            if gt_boxes.shape[0] > 0 and pred_boxes.shape[0] > 0:
                builder.add(pred_boxes, gt_boxes)
                
            if image:
                image.close()
                del image
            
            gc.collect()
        
        # Calculate metrics
        metrics = builder.value(iou_thresholds=[round(x, 2) for x in np.arange(0.5, 1.0, 0.05)])
        

        # Extract metrics
        metrics_dict = {
            "dataset_test/map50-95": metrics["mAP"],
            "dataset_test/map50": metrics[0.5][0]["ap"],
            "dataset_test/map75": metrics[0.75][0]["ap"],
            "dataset_test/precision": metrics[0.75][0]["precision"].mean(),
            "dataset_test/recall": metrics[0.75][0]["recall"].mean()
        }

        self._log_to_wandb(metrics_dict, wandb_run)
        self._display_metrics(metrics_dict)
        self._finish_wandb(wandb_run)
        return metrics_dict
    

    def predict(self, data_path, output_dir, save_image=True):
        """
        Predict on all images in a directory and save results as ALTO XML files.
        Based on code by Thibault Clérice (https://github.com/ponteineptique/yolalto).
        
        Args:
            output_dir: Directory to save ALTO XML files
            save_image: option to save the image along with the output or not
        """
        if self.model_loaded != "trained":
           self.load("trained")
        
        # Find all images in the corpus directory
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(data_path, ext)))
        
        if not image_paths:
            raise ValueError(f"No images found in {data_path}")
        # Process images in batches
        batch_size = 4  # Default batch size
        num_batches = len(image_paths) // batch_size + (1 if len(image_paths) % batch_size else 0)
        
        print(f"Processing {len(image_paths)} images in {num_batches} batches...")
        
        for batch_idx in range(num_batches):
            batch = image_paths[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            
            try:
                # Run model on batch
                results = self.model.predict(batch, save=False, verbose=False)
                
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
                
                    if save_image:
                        image_filename = os.path.basename(image_path)
                        image_output_path = os.path.join(output_dir, image_filename)
                        shutil.copy2(image_path, image_output_path)

            except Exception as e:
                print(f"Error processing batch: {e}")
                import traceback
                traceback.print_exc()

        return results