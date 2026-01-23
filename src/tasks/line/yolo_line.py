from src.tasks.line.base_line import BaseLine
import os
import glob
import gc
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path

from ultralytics import YOLO, settings

from src.alto.alto_lines import add_lines_to_alto


class YoloLineTask(BaseLine):
    """
    Line segmentation using YOLO.
    Detects text lines as bounding boxes and converts them to baseline format.
    """
    
    def __init__(self, config):
        """
        Initialize YOLO line segmentation model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        settings.update({"wandb": config.get('use_wandb', True)})
        
        self.name = "Line Segmentation (YOLO)"
        self.wandb_project = "LS-comparison"
        
        self.model_loaded = None
        self.model = None
    
    def load(self, mode="trained"):
        """
        Load the YOLO model from weights file.
        
        Args:
            mode: "trained" or "pretrained"
        """
        if mode == "pretrained":
            if not os.path.exists(self.config["pretrained_w"]):
                raise FileNotFoundError(
                    f"Pretrained weights not found: {self.config['pretrained_w']}"
                )
            model_path = self.config["pretrained_w"]
            self.model_loaded = "pretrained"
        
        elif mode == "trained":
            if not self.config.get("model_path"):
                raise FileNotFoundError(
                    f"Trained weights not found: {self.config['model_path']}"
                )
            model_path = self.config["model_path"]
            self.model_loaded = "trained"
        
        self.model = YOLO(model_path)
        self.to_device()
    
    def train(self, data_path=None, seed=42):
        """
        Train the YOLO line segmentation model.
        
        Args:
            data_path: Path to training data (YOLO format)
            seed: Random seed
        """
        if self.model_loaded != "pretrained":
            self.load("pretrained")
        
        training_data = data_path or self.config.get("data_path")
        if not training_data:
            raise ValueError(
                "No training data provided in config or as argument"
            )
        
        save_name = (f"{self.name}_{self.config['img_size']}px_"
                    f"{self.config['batch_size']}bs_"
                    f"{self.config['epochs']}e")
        
        self.model.train(
            data=training_data,
            project='LS-training',
            imgsz=self.config["img_size"],
            batch=self.config["batch_size"],
            epochs=self.config["epochs"],
            name=save_name,
            device=self.config["device"],
            seed=seed
        )
    
    def _yolo_box_to_line(self, box, image_width, image_height):
        """
        Convert YOLO detection box to line format (baseline + boundary).
        
        Args:
            box: YOLO box coordinates [x1, y1, x2, y2]
            image_width: Image width
            image_height: Image height
            
        Returns:
            Dictionary with baseline and boundary
        """
        x1, y1, x2, y2 = map(int, box)
        
        # Create baseline at middle of box (horizontal line)
        baseline_y = (y1 + y2) // 2
        baseline = [[x1, baseline_y], [x2, baseline_y]]
        
        # Create boundary (rectangle around the line)
        boundary = [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ]
        
        return {
            'baseline': baseline,
            'boundary': boundary,
            'id': f'line_{id(box)}'
        }
    
    def _process_batch(self, file_paths, source_dir, output_dir, save_image=True):
        """
        Process a batch of images for YOLO line segmentation.
        
        Args:
            file_paths: List of image paths to process
            source_dir: Source directory (for finding ALTO if exists)
            output_dir: Directory to save ALTO XML files
            save_image: Whether to copy images to output
            
        Returns:
            List of prediction results
        """
        print(f"  Processing {len(file_paths)} images...")
        
        results = []
        batch_size = self.config.get('batch_size', 4)
        num_batches = (len(file_paths) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            batch = file_paths[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            
            try:
                # Run YOLO on batch
                batch_results = self.model.predict(
                    batch, 
                    save=False, 
                    verbose=False,
                    imgsz=self.config.get('img_size', 640)
                )
                
                # Process each image result
                for image_path, result in zip(batch, batch_results):
                    image = Image.open(image_path)
                    image_size = image.size
                    
                    # Convert YOLO detections to lines
                    lines = []
                    for box in result.boxes:
                        xyxy = box.xyxy[0].tolist()
                        line = self._yolo_box_to_line(
                            xyxy, 
                            image_size[0], 
                            image_size[1]
                        )
                        lines.append(line)
                    
                    # Find or create ALTO file
                    base_name = Path(image_path).stem
                    output_path = os.path.join(output_dir, f"{base_name}.xml")
                    
                    # Check for existing ALTO with layout
                    existing_alto = os.path.join(source_dir, f"{base_name}.xml")
                    if os.path.exists(existing_alto):
                        # Use existing structure
                        add_lines_to_alto(lines, output_path, existing_alto)
                    else:
                        # Create new ALTO
                        from src.alto.yolalto import create_alto_xml, save_alto_xml
                        
                        # Create minimal layout structure
                        detections = [{
                            'label': 'MainZone',
                            'bbox': [0, 0, image_size[0], image_size[1]]
                        }]
                        alto = create_alto_xml(detections, image_path, image_size)
                        save_alto_xml(alto, output_path)
                        
                        # Add lines
                        add_lines_to_alto(lines, output_path, output_path)
                    
                    results.append(lines)
                    
                    # Copy image if requested
                    if save_image:
                        image_output = os.path.join(
                            output_dir, 
                            os.path.basename(image_path)
                        )
                        shutil.copy2(image_path, image_output)
                    
                    image.close()
                    
            except Exception as e:
                print(f"  Error processing batch: {e}")
                import traceback
                traceback.print_exc()
            
            finally:
                gc.collect()
        
        return results