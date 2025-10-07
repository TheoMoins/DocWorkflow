from src.tasks.htr.base_htr import BaseHTR
import os
import glob
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import shutil
from lxml import etree as ET
import numpy as np

from kraken import rpred
from kraken.lib.models import load_any
from kraken.containers import Segmentation, BaselineLine

from src.alto.alto_lines import extract_lines_from_alto


class KrakenHTRTask(BaseHTR):
    """
    HTR implementation using Kraken.
    Requires pre-segmented lines from ALTO XML files.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "HTR (Kraken)"
    
    def load(self):
        """
        Load the Kraken HTR model.
        """
        model_path = self.config.get('model_path')
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"HTR model not found: {model_path}")
        
        self.model = load_any(model_path)
        self.to_device()
    
    def _recognize_text(self, image_path, alto_path):
        """
        Recognize text by extracting lines from ALTO and processing them.
        
        Args:
            image_path: Path to the image file
            alto_path: Path to the ALTO XML file
            
        Returns:
            List of recognized text for each line
        """
        results = []
        
        # Parse ALTO to extract lines
        tree = ET.parse(alto_path)
        root = tree.getroot()
        ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
        
        # Extract all lines with baselines
        lines_data = []
        for textline in root.findall('.//alto:TextLine', ns):
            baseline_str = textline.get('BASELINE', '')
            if baseline_str:
                coords = baseline_str.split()
                baseline_points = []
                for i in range(0, len(coords), 2):
                    if i + 1 < len(coords):
                        baseline_points.append([float(coords[i]), float(coords[i+1])])
                
                if len(baseline_points) >= 2:
                    # Get polygon if available
                    polygon = textline.find('.//alto:Polygon', ns)
                    boundary = None
                    if polygon is not None:
                        points_str = polygon.get('POINTS', '')
                        if points_str:
                            coords = points_str.split()
                            boundary_points = []
                            for i in range(0, len(coords), 2):
                                if i + 1 < len(coords):
                                    boundary_points.append([float(coords[i]), float(coords[i+1])])
                            if boundary_points:
                                boundary = boundary_points
                    
                    # Create boundary from baseline if not available
                    if boundary is None:
                        baseline_np = np.array(baseline_points)
                        min_x = baseline_np[:, 0].min()
                        max_x = baseline_np[:, 0].max()
                        min_y = baseline_np[:, 1].min() - 10
                        max_y = baseline_np[:, 1].max() + 10
                        boundary = [[min_x, min_y], [max_x, min_y], 
                                  [max_x, max_y], [min_x, max_y]]
                    
                    lines_data.append({
                        'baseline': baseline_points,
                        'boundary': boundary
                    })
        
        if not lines_data:
            print(f"  No lines with baseline found in {alto_path}")
            return []
        
        # Load image
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create BaselineLine objects for Kraken
        kraken_lines = []
        for line_data in lines_data:
            line = BaselineLine(
                id=f"line_{len(kraken_lines)}",
                baseline=line_data['baseline'],
                boundary=line_data['boundary']
            )
            kraken_lines.append(line)
        
        # Create Segmentation object
        segmentation = Segmentation(
            type='baselines',
            imagename=image_path,
            text_direction='horizontal-lr',
            script_detection=False,
            lines=kraken_lines,
            regions={}
        )
        
        # Perform recognition
        predictions = list(rpred.rpred(
            self.model,
            image,
            segmentation,
            pad=16,
            bidi_reordering=True
        ))
        
        # Convert results
        for pred in predictions:
            text = ''
            confidence = 0.0
            
            if hasattr(pred, 'prediction'):
                text = pred.prediction
            
            if hasattr(pred, 'confidences') and pred.confidences:
                confidence = sum(pred.confidences) / len(pred.confidences)
            
            results.append({
                'text': text,
                'confidence': confidence
            })
        
        image.close()
        return results
    
    def _add_text_to_alto(self, alto_path, texts, output_path):
        """
        Add recognized text to existing ALTO XML file.
        
        Args:
            alto_path: Path to input ALTO file
            texts: List of recognized texts
            output_path: Path to save modified ALTO
        """
        tree = ET.parse(alto_path)
        root = tree.getroot()
        ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
        
        text_lines = root.findall('.//alto:TextLine', ns)
        
        for line, text_data in zip(text_lines, texts):
            if text_data and 'text' in text_data:
                # Remove existing String elements
                for string_elem in line.findall('alto:String', ns):
                    line.remove(string_elem)
                
                # Add new String element
                string_elem = ET.SubElement(line, f"{{{ns['alto']}}}String")
                string_elem.set('CONTENT', text_data['text'])
                string_elem.set('WC', str(text_data.get('confidence', 0.0)))
        
        tree.write(output_path, pretty_print=True, 
                  xml_declaration=True, encoding="UTF-8")
    
    def predict(self, data_path, output_dir, save_image=True):
        """
        Perform HTR on ALTO files with line segmentation.
        
        Args:
            data_path: Directory containing images and ALTO files
            output_dir: Directory to save ALTO XML files with text
            save_image: Whether to copy images to output directory
            
        Returns:
            List of prediction results
        """
        if not self.model:
            self.load()
        
        alto_files = sorted(glob.glob(os.path.join(data_path, "*.xml")))
        
        if not alto_files:
            raise ValueError(f"No ALTO XML files found in {data_path}")
        
        print(f"Processing {len(alto_files)} ALTO files...")
        
        results = []
        for alto_path in tqdm(alto_files, desc="Recognizing text", unit="page"):
            try:
                image_path, lines, _ = extract_lines_from_alto(alto_path)
                
                if not os.path.exists(image_path):
                    print(f"Warning: Image {image_path} not found")
                    continue
                
                if not lines:
                    print(f"Warning: No lines found in {alto_path}")
                    continue
                
                recognized_texts = self._recognize_text(image_path, alto_path)
                
                output_path = os.path.join(output_dir, os.path.basename(alto_path))
                
                if not os.path.exists(output_path):
                    shutil.copy2(alto_path, output_path)
                
                self._add_text_to_alto(output_path, recognized_texts, output_path)
                
                results.append({
                    'file': alto_path,
                    'texts': recognized_texts
                })
                
                if save_image:
                    image_output = os.path.join(output_dir, os.path.basename(image_path))
                    if not os.path.exists(image_output):
                        shutil.copy2(image_path, image_output)
                
            except Exception as e:
                print(f"Error processing {alto_path}: {e}")
                import traceback
                traceback.print_exc()
        
        return results