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
import threading

from kraken.lib.vgsl import TorchVGSLModel
from kraken.kraken import SEGMENTATION_DEFAULT_MODEL
from kraken.lib.xml import XMLPage
from yaltai.models.krakn import segment as line_segment

from src.utils.utils import IGNORED_ZONE_TYPES
from src.utils.memory_monitor import check_memory_safe, force_garbage_collection, get_memory_usage_percent
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
            List of predicted lines
        """
        result = [None]
        is_timeout = [False]
        
        def process_image():
            try:
                _, _, all_regions = extract_lines_from_alto(file_name)

                # Only keep mainzones
                alto_regions = {
                    region_type: polygons 
                    for region_type, polygons in all_regions.items() 
                    if region_type not in IGNORED_ZONE_TYPES
                }                
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
                
                if not is_timeout[0]:
                    result[0] = lines
            except Exception as e:
                print(f"  Error during line segmentation: {e}")
                if not is_timeout[0]:
                    result[0] = []
        
        # TODO : Mauvaise gestion du timeout dans la partie eval (ou dans score, en tout cas les deux gestions sont différentes)
        # Créer et démarrer le thread de traitement
        thread = threading.Thread(target=process_image)
        thread.daemon = True
        thread.start()
        thread.join(timeout=90)
        
        if thread.is_alive():
            is_timeout[0] = True
            print(f"  Timeout for {file_name}: Interruption after 60 seconds")
            return []
        
        return result[0] if result[0] is not None else []


    def _score_batch(self, pred_files, gt_files, pred_dir, gt_dir):
        """
        Score line predictions for a batch of files.
        
        Args:
            pred_files: List of prediction ALTO file paths
            gt_files: List of ground truth ALTO file paths
            pred_dir: Prediction directory
            gt_dir: Ground truth directory
            
        Returns:
            Tuple of (metrics_dict, page_scores)
        """
        
        # Initialize metrics builder
        builder = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=1)
        
        page_scores = []
        
        # Process files
        for pred_file, gt_file in tqdm(zip(pred_files, gt_files), total=len(pred_files),
                                        desc="  Scoring", unit="page"):
            image_path, gt_lines, _ = extract_lines_from_alto(gt_file)
            if not gt_lines:
                print(f"  Warning: No lines in {Path(gt_file).name}")
                continue
            
            _, pred_lines, _ = extract_lines_from_alto(pred_file)
            if not pred_lines:
                print(f"  Warning: No predictions in {Path(pred_file).name}")
                continue
            
            try:
                image = Image.open(image_path)
                image_size = image.size
            except Exception as e:
                print(f"  Error opening image {image_path}: {e}")
                continue
            
            # Convert to boxes
            gt_boxes = convert_lines_to_boxes(gt_lines, image_size, is_gt=True)
            pred_boxes = convert_lines_to_boxes(pred_lines, image_size, is_gt=False)
            
            if gt_boxes.shape[0] > 0 and pred_boxes.shape[0] > 0:
                builder.add(pred_boxes, gt_boxes)
                
                # Store per-page info
                page_scores.append({
                    'page': Path(gt_file).stem,
                    'gt_lines': len(gt_lines),
                    'pred_lines': len(pred_lines),
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
        """Line scores XML files."""
        return ['*.xml']
    
    def _process_batch(self, file_paths, source_dir, output_dir, save_image=True):
        """
        Process a batch of images for line segmentation.
        
        Args:
            file_paths: List of image paths to process
            source_dir: Source directory (pour trouver les ALTO XML correspondants)
            output_dir: Directory to save ALTO XML files with lines
            save_image: Whether to copy images to output
            
        Returns:
            List of prediction results
        """
        print(f"  Processing {len(file_paths)} images...")

        results = []
        for image_path in tqdm(file_paths, desc="  Predicting lines", unit="image"):
            # Check memory before processing each image
            is_safe, mem_msg = check_memory_safe(min_available_gb=2.0, max_usage_percent=85)
            if not is_safe:
                print(f"\nWarning: {mem_msg}")
                print("Forcing garbage collection...")
                force_garbage_collection()
                
                # Re-check after GC
                is_safe, mem_msg = check_memory_safe(min_available_gb=1.0, max_usage_percent=95)
                if not is_safe:
                    print(f"Memory still critical after GC: {mem_msg}")
                    print("Skipping remaining images to prevent system crash.")
                    break
            try:
                # Check for corresponding ALTO XML file with layout regions
                alto_path = os.path.join(source_dir, Path(image_path).with_suffix('.xml').name)
                
                # Check if ALTO file exists and contains layout regions
                has_layout = False
                if os.path.exists(alto_path):
                    _, _, regions = extract_lines_from_alto(alto_path)
                    has_layout = bool(regions)
                
                if not has_layout:
                    print(f"  Warning: No layout regions found for {image_path}. Results may be suboptimal.")
                
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
                print(f"  Error processing {image_path}: {e}")
                import traceback
                traceback.print_exc()

            finally:
                # Aggressive memory cleanup after each image
                if 'image' in locals() and image is not None:
                    image.close()
                    del image
                
                # Force garbage collection every image
                gc.collect()
                
                # Clear PyTorch cache if using GPU
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
        
        return results