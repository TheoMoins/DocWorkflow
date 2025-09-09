from core.base_model import BaseModel
import torch
import os
import glob
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import shutil
from lxml import etree as ET

from kraken import rpred
from kraken.lib.models import load_any
from kraken.lib.xml import XMLPage
from kraken.containers import Segmentation
from kraken import rpred
import numpy as np

from core.utils import extract_lines_from_alto

class HTRModel(BaseModel):
    """
    Class for HTR (Handwritten Text Recognition) models.
    """
    
    def __init__(self, config):
        """
        Initialize the HTR model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)

        self.wandb_project = "HTR-comparison"
    
    def load(self):
        """
        Load the model from a weights file.
        """
        model_path = self.config.get('model_path')
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"HTR model not found: {model_path}")
        
        self.model = load_any(model_path)
        self.to_device()
    
    def train(self, data_path=None, **kwargs):
        """
        Train the HTR model.
        
        Args:
            data_path: Path to training data
            **kwargs: Additional training arguments
        """
        # TODO
        print("Training for HTR models is not yet implemented.")
    
    def _recognize_text(self, image, lines):
        """
        Recognize text in the given lines.
        
        Args:
            image: PIL Image object
            lines: List of line dictionaries with baselines
            
        Returns:
            List of recognized text for each line
        """
        
        results = []
        
        # Create a Segmentation object
        segmentation = Segmentation(
            type='baselines',
            imagename=None,
            text_direction='horizontal-lr',
            script_detection=False,
            lines=lines,
            regions={}
        )
        
        # Run recognition
        pred_it = rpred.rpred(self.model, image, segmentation)
        
        for pred in pred_it:
            results.append({
                'text': pred.prediction,
                'confidence': sum(pred.confidences) / len(pred.confidences) if pred.confidences else 0.0
            })

        
        return results
    
    def _add_text_to_alto(self, alto_path, texts, output_path):
        """
        Add recognized text to ALTO XML file.
        
        Args:
            alto_path: Path to input ALTO file
            texts: List of recognized texts
            output_path: Path to save modified ALTO
        """
        tree = ET.parse(alto_path)
        root = tree.getroot()
        ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
        
        # Find all TextLine elements
        text_lines = root.findall('.//alto:TextLine', ns)
        
        # Add text content to each line
        for line, text_data in zip(text_lines, texts):
            if text_data and 'text' in text_data:
                # Remove existing String elements if any
                for string_elem in line.findall('alto:String', ns):
                    line.remove(string_elem)
                
                # Add new String element with recognized text
                string_elem = ET.SubElement(line, f"{{{ns['alto']}}}String")
                string_elem.set('CONTENT', text_data['text'])
                string_elem.set('WC', str(text_data.get('confidence', 0.0)))
        
        # Save modified ALTO
        tree.write(output_path, pretty_print=True, 
                  xml_declaration=True, encoding="UTF-8")
    
    def _compute_metrics(self, test_path, is_corpus=False):
        """
        Compute metrics for the HTR model.
        
        Args:
            test_path: Path to test data
            is_corpus: Whether this is evaluation on corpus data
            
        Returns:
            Dictionary of evaluation metrics
        """
        # TODO: Implement CER/WER metrics calculation
        print("HTR metrics computation not yet implemented")
        return {"CER": 0.0, "WER": 0.0}
    
    def predict(self, output_dir, save_image=False):
        """
        Perform HTR on ALTO files with line segmentation.
        
        Args:
            output_dir: Directory to save ALTO XML files with text
            save_image: Whether to copy images to output directory
            
        Returns:
            Prediction results
        """
        if not self.model:
            self.load()
        
        pred_path = self.config.get("pred_path")
        if not pred_path:
            raise ValueError("No pred_path specified in config")
        
        # Find all ALTO XML files
        alto_files = sorted(glob.glob(os.path.join(pred_path, "*.xml")))
        
        if not alto_files:
            raise ValueError(f"No ALTO XML files found in {pred_path}")
        
        print(f"Processing {len(alto_files)} ALTO files...")
        
        results = []
        for alto_path in tqdm(alto_files, desc="Recognizing text", unit="page"):
            try:
                # Extract image path and lines from ALTO
                image_path, lines, _ = extract_lines_from_alto(alto_path)
                
                if not os.path.exists(image_path):
                    print(f"Warning: Image {image_path} not found")
                    continue
                
                if not lines:
                    print(f"Warning: No lines found in {alto_path}")
                    continue
                
                # Load image
                image = Image.open(image_path)
                
                # Recognize text
                recognized_texts = self._recognize_text(image, lines)
                
                # Save ALTO with recognized text
                output_path = os.path.join(output_dir, 
                                         os.path.basename(alto_path))
                
                # Copy original ALTO first
                if not os.path.exists(output_path) and os.path.exists(alto_path):
                    shutil.copy2(alto_path, output_path)
                
                # Add recognized text
                self._add_text_to_alto(output_path, recognized_texts, output_path)
                
                results.append({
                    'file': alto_path,
                    'texts': recognized_texts
                })
                
                # Copy image if requested
                if save_image:
                    image_output = os.path.join(output_dir, 
                                               os.path.basename(image_path))
                    shutil.copy2(image_path, image_output)
                
                image.close()
                
            except Exception as e:
                print(f"Error processing {alto_path}: {e}")
                import traceback
                traceback.print_exc()
        
        return results