from src.tasks.base_tasks import BaseTask
import torch
import os
from tqdm import tqdm
from PIL import Image
import glob
import numpy as np
from mean_average_precision import MetricBuilder
from pathlib import Path
import gc
import shutil
import tempfile

import threading

from kraken.lib.vgsl import TorchVGSLModel
from kraken.kraken import SEGMENTATION_DEFAULT_MODEL
from kraken.lib.xml import XMLPage
from yaltai.models.krakn import segment as line_segment

from src.alto.alto_lines import extract_lines_from_alto, convert_lines_to_boxes, add_lines_to_alto

class KrakenLineTask(BaseTask):
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
        self.name = "Line Segmentation (Kraken)"
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
        
        # TODO : Mauvaise gestion du timeout dans la partie eval (ou dans score, en tout cas les deux gestions sont différentes)
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


    def score(self, pred_path, gt_path):
        """
        Calculate scores between prediction ALTO files and ground truth ALTO files.
        
        Args:
            pred_path: Path to directory containing prediction ALTO files
            gt_path: Path to directory containing ground truth ALTO files
            
        Returns:
            Dictionary of evaluation metrics
        """
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
            
            # Extract ground truth lines and image path
            image_path, gt_lines, _ = extract_lines_from_alto(gt_file)
            if not gt_lines:
                print(f"Warning: No lines found in {gt_file}")
                continue
            
            # Extract predicted lines
            _, pred_lines, _ = extract_lines_from_alto(pred_file)
            if not pred_lines:
                print(f"Warning: No predicted lines found in {pred_file}")
                continue
            
            try:
                image = Image.open(image_path)
                image_size = image.size
            except Exception as e:
                print(f"Error opening image {image_path}: {e}")
                continue
            
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
        metrics_dict = {
            "dataset_test/map50-95": metrics["mAP"],
            "dataset_test/map50": metrics[0.5][0]["ap"],
            "dataset_test/map75": metrics[0.75][0]["ap"],
            "dataset_test/precision": metrics[0.75][0]["precision"].mean(),
            "dataset_test/recall": metrics[0.75][0]["recall"].mean()
        }

        self._display_metrics(metrics_dict)

        return metrics_dict

    
    def predict(self, data_path, output_dir, save_image=True):
        """
        Perform prediction on an image.
        
        Args:
            output_dir: Directory to save ALTO XML files       
            save_image: if True: copy the image in the output_dir
        Returns:
            Prediction results
        """
        if not self.model:
            self.load()

        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(data_path, ext)))
        
        if not image_paths:
            raise ValueError(f"No images found in {data_path}")
        
        print(f"Processing {len(image_paths)} images...")

        results = []
        for image_path in tqdm(image_paths, desc="Predicting lines", unit="image"):
            try:
                # Check for corresponding ALTO XML file with layout regions
                alto_path = os.path.join(data_path, Path(image_path).with_suffix('.xml').name)
                
                # Check if ALTO file exists and contains layout regions
                has_layout = False
                if os.path.exists(alto_path):
                    _, _, regions = extract_lines_from_alto(alto_path)
                    has_layout = bool(regions)
                
                if not has_layout:
                    print(f"Warning: No layout regions found for {image_path}. Results may be suboptimal.")
                
                image = Image.open(image_path)
                
                # Predict lines
                if has_layout:
                    predicted_lines = self._predict_page(image, alto_path)
                else:
                    segmentation_result = line_segment(
                        image,
                        text_direction=self.config.get("text_direction", "horizontal-lr"),
                        model=self.model,
                        device=self.device,
                        regions={},
                        ignore_lignes=False
                    )
                    predicted_lines = [{'baseline': line.baseline, 'boundary': line.boundary, 'id': line.id}
                                    for line in segmentation_result.lines]
                
                # Generate output ALTO XML file path
                output_path = os.path.join(output_dir, Path(image_path).with_suffix('.xml').name)

                if not os.path.exists(output_path) and os.path.exists(alto_path):
                    shutil.copy2(alto_path, output_path) 

                add_lines_to_alto(predicted_lines, output_path, alto_path)
                
                results.append(predicted_lines)
                
                if save_image:
                    image_filename = os.path.basename(image_path)
                    image_output_path = os.path.join(output_dir, image_filename)
                    shutil.copy2(image_path, image_output_path)

                # Close the image
                image.close()
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                import traceback
                traceback.print_exc()
        
        return results
