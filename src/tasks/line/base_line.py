from src.tasks.base_tasks import BaseTask
from abc import abstractmethod
import numpy as np
from mean_average_precision import MetricBuilder
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import gc
import warnings

from src.alto.alto_lines import read_lines_geometry, convert_lines_to_boxes, read_lines_for_zonemap
from src.utils.zonemap import compute_zonemap_page, accumulate_zonemap_stats, finalize_zonemap_metrics


class BaseLine(BaseTask):
    """
    Base class for line segmentation tasks.
    Provides common scoring and evaluation methods.
    """
    
    def __init__(self, config):
        """
        Initialize the line segmentation base class.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        self.text_direction = config.get('text_direction', 'horizontal-lr')
    
    @abstractmethod
    def load(self):
        """Load the model. Must be implemented by subclasses."""
        pass
    
    def train(self, data_path=None, **kwargs):
        """
        Train the line segmentation model. 
        Default implementation prints a message.
        Can be overridden by subclasses that support training.
        """
        print(f"Training for {self.name} is not yet implemented.")
    
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
        warnings.filterwarnings('ignore', category=FutureWarning, 
                              module='mean_average_precision')
        
        # Initialize metrics builder
        builder = MetricBuilder.build_evaluation_metric(
            "map_2d", async_mode=False, num_classes=1
        )
        
        page_scores = []
        
        # Process files
        for pred_file, gt_file in tqdm(zip(pred_files, gt_files), 
                                       total=len(pred_files),
                                       desc="  Scoring", unit="page"):
            image_path, gt_lines, _ = read_lines_geometry(gt_file)
            if not gt_lines:
                print(f"  Warning: No lines in {Path(gt_file).name}")
                continue
            
            pred_lines = read_lines_for_zonemap(pred_file)
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
        metrics = builder.value(
            iou_thresholds=[round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]
        )
        
        metrics_dict = {
            "dataset_test/map50-95": metrics["mAP"],
            "dataset_test/map50": metrics[0.5][0]["ap"],
            "dataset_test/map75": metrics[0.75][0]["ap"],
            "dataset_test/precision": metrics[0.75][0]["precision"].mean() if len(metrics[0.75][0]["precision"]) > 0 else 0.0,
            "dataset_test/recall": metrics[0.75][0]["recall"].mean() if len(metrics[0.75][0]["recall"]) > 0 else 0.0
        }

        # ZoneMapAlt — detection metric (page-by-page accumulation avoids false
        # cross-page polygon intersections since all pages share origin (0,0))
        zm_accumulated = None
        for pred_file, gt_file in zip(pred_files, gt_files):
            gt_lines = read_lines_for_zonemap(gt_file)
            if not gt_lines:
                continue
            dt_lines = read_lines_for_zonemap(pred_file)
            if not dt_lines:
                continue
            page_stats = compute_zonemap_page(gt_lines, dt_lines, with_recognition=False)
            zm_accumulated = accumulate_zonemap_stats(zm_accumulated, page_stats)

        if zm_accumulated is not None:
            metrics_dict.update(finalize_zonemap_metrics(zm_accumulated))

        return metrics_dict, page_scores

    def _get_score_file_extensions(self):
        """Line segmentation scores XML files."""
        return ['*.xml']