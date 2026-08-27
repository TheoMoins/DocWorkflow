from src.tasks.base_tasks import BaseTask
from abc import abstractmethod
import numpy as np
from mean_average_precision import MetricBuilder
from tqdm import tqdm
from pathlib import Path
import warnings

from src.alto.alto_lines import read_lines, convert_lines_to_boxes, DEFAULT_BASELINE_RATIO
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
        # Vertical placement of the baseline inside a line polygon.
        self.baseline_ratio = config.get('baseline_ratio', DEFAULT_BASELINE_RATIO)
        # Lines falling outside every layout zone: kept as single-line pseudo-zones
        # (default, production-safe) or dropped. Dropping constrains the model to the
        # layout the way Kraken is constrained by construction, which is the
        # symmetric protocol for a benchmark.
        self.orphan_policy = (
            "drop" if config.get('restrict_to_layout', False) else "pseudo_block"
        )
    
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

        Ground truth and predictions go through the *same* reader (read_lines) and the
        *same* box conversion, for every metric. Boxes stay in pixel coordinates and
        carry the detector confidence, so the precision/recall curve the AP integrates
        is the real one.

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
        zm_accumulated = None
        
        # Process files
        for pred_file, gt_file in tqdm(zip(pred_files, gt_files), 
                                       total=len(pred_files),
                                       desc="  Scoring", unit="page"):
            gt_lines = read_lines(gt_file)
            if not gt_lines:
                print(f"  Warning: No lines in {Path(gt_file).name}")
                continue
            
            pred_lines = read_lines(pred_file)
            if not pred_lines:
                print(f"  Warning: No predictions in {Path(pred_file).name}")
                continue
            
            # Pixel coordinates: no image needed, and nothing is quantised away.
            gt_boxes = convert_lines_to_boxes(gt_lines, is_gt=True)
            pred_boxes = convert_lines_to_boxes(pred_lines, is_gt=False)
            
            if gt_boxes.shape[0] > 0 and pred_boxes.shape[0] > 0:
                builder.add(pred_boxes, gt_boxes)
                
                # Store per-page info
                page_scores.append({
                    'page': Path(gt_file).stem,
                    'gt_lines': len(gt_lines),
                    'pred_lines': len(pred_lines),
                })
            
            # ZoneMapAlt — detection metric (page-by-page accumulation avoids false
            # cross-page polygon intersections since all pages share origin (0,0))
            page_stats = compute_zonemap_page(gt_lines, pred_lines, with_recognition=False)
            zm_accumulated = accumulate_zonemap_stats(zm_accumulated, page_stats)
        
        # Calculate global metrics
        metrics = builder.value(
            iou_thresholds=[round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]
        )
        
        # `precision` et `recall` sont les tableaux **cumulés** de la courbe
        # précision-rappel, un point par détection classée par confiance
        # décroissante. Leur moyenne n'est pas un taux : sur un détecteur parfait
        # elle donne un rappel de 0,55, et son plafond tend vers 0,5 quand le
        # nombre de détections croît. On prend le dernier point, c'est-à-dire le
        # point d'opération obtenu en gardant toutes les détections émises par le
        # modèle (YOLO ayant déjà appliqué son propre seuil de confiance).
        prec_curve = metrics[0.75][0]["precision"]
        rec_curve = metrics[0.75][0]["recall"]
        precision = float(prec_curve[-1]) if len(prec_curve) > 0 else 0.0
        recall = float(rec_curve[-1]) if len(rec_curve) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # float() explicite : mean_average_precision renvoie des np.float32, que
        # l'agrégation inter-documents et les exports traitent moins bien.
        metrics_dict = {
            "dataset_test/map50-95": float(metrics["mAP"]),
            "dataset_test/map50": float(metrics[0.5][0]["ap"]),
            "dataset_test/map75": float(metrics[0.75][0]["ap"]),
            "dataset_test/precision": precision,
            "dataset_test/recall": recall,
            "dataset_test/f1": f1,
        }

        if zm_accumulated is not None:
            metrics_dict.update(finalize_zonemap_metrics(zm_accumulated))

        return metrics_dict, page_scores

    def _get_score_file_extensions(self):
        """Line segmentation scores XML files."""
        return ['*.xml']