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

        self.name = "Layout_YOLO"

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

        self.model.train(
            data=training_data, 
            project='LA-training', 
            imgsz=self.config["img_size"],
            batch=self.config["batch_size"],
            epochs=self.config["epochs"], 
            name=save_name,
            device=self.config["device"],
            seed=seed
        )

    def _score_batch(self, pred_files, gt_files, pred_dir, gt_dir):
        """
        Score layout predictions for a batch of files.
        
        Args:
            pred_files: List of prediction ALTO file paths
            gt_files: List of ground truth ALTO file paths
            pred_dir: Prediction directory
            gt_dir: Ground truth directory
            
        Returns:
            Tuple of (metrics_dict, page_scores)
        """
        warnings.filterwarnings('ignore', category=FutureWarning, module='mean_average_precision')
        
        # Initialize metrics builder
        builder = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=1)
        
        page_scores = []
        
        # Process files
        for pred_file, gt_file in tqdm(zip(pred_files, gt_files), total=len(pred_files), 
                                        desc="  Scoring", unit="page"):
            # Extract ground truth zones and image path
            image_path, gt_zones = extract_zones_from_alto(gt_file)
            if not gt_zones:
                print(f"  Warning: No zones in {Path(gt_file).name}")
                continue
            
            # Extract predicted zones
            _, pred_zones = extract_zones_from_alto(pred_file)
            if not pred_zones:
                print(f"  Warning: No predictions in {Path(pred_file).name}")
                continue
            
            try:
                image = Image.open(image_path)
                image_size = image.size
            except Exception as e:
                print(f"  Error opening image {image_path}: {e}")
                continue
            
            # Convert to boxes
            gt_boxes = convert_zones_to_boxes(gt_zones, image_size, is_gt=True)
            pred_boxes = convert_zones_to_boxes(pred_zones, image_size, is_gt=False)
            
            if gt_boxes.shape[0] > 0 and pred_boxes.shape[0] > 0:
                builder.add(pred_boxes, gt_boxes)
                
                # Store per-page info
                page_scores.append({
                    'page': Path(gt_file).stem,
                    'gt_zones': len(gt_zones),
                    'pred_zones': len(pred_zones),
                })
            
            if image:
                image.close()
                del image
            
            gc.collect()
        
        # Calculate global metrics
        metrics = builder.value(iou_thresholds=[round(x, 2) for x in np.arange(0.5, 1.0, 0.05)])
        
        metrics_dict = {
            "dataset_test/map50-95": metrics["mAP"],
            "dataset_test/map50": metrics[0.5][0]["ap"],
            "dataset_test/map75": metrics[0.75][0]["ap"],
            "dataset_test/precision": metrics[0.75][0]["precision"].mean(),
            "dataset_test/recall": metrics[0.75][0]["recall"].mean()
        }
        
        return metrics_dict, page_scores

    def _get_score_file_extensions(self):
        """Layout scores XML files."""
        return ['*.xml']
    

    def _process_batch(self, file_paths, source_dir, output_dir, save_image=True):
        """
        Process a batch of images for layout segmentation.
        
        Args:
            file_paths: List of image paths to process
            source_dir: Source directory (unused for layout, mais requis par l'interface)
            output_dir: Directory to save ALTO XML files
            save_image: Whether to copy images to output
            
        Returns:
            List of result paths
        """
        # Process images in batches
        batch_size = 4
        num_batches = len(file_paths) // batch_size + (1 if len(file_paths) % batch_size else 0)
        
        print(f"  Processing {len(file_paths)} images in {num_batches} batches...")
        
        results = []
        
        for batch_idx in range(num_batches):
            batch = file_paths[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            
            try:
                # Run model on batch
                batch_results = self.model.predict(batch, save=False, verbose=False)
                
                # Parse results
                batch_detections = parse_yolo_results(batch_results)
                
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
                    
                    results.append(output_path)

            except Exception as e:
                print(f"  Error processing batch: {e}")
                import traceback
                traceback.print_exc()

        return results