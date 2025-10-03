from src.tasks.base_tasks import BaseTask
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
from kraken.containers import Segmentation, BaselineLine
from kraken import rpred

import numpy as np

from src.alto.alto_lines import extract_lines_from_alto

class KrakenHTRTask(BaseTask):
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

        self.name = "HTR (Kraken)"

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
    
    def _recognize_text(self, image_path, alto_path):
        """
        Recognize text by extracting lines from ALTO and processing them directly.
        
        Args:
            image_path: Path to the image file
            alto_path: Path to the ALTO XML file
            
        Returns:
            List of recognized text for each line
        """
        
        results = []
        
        # try:
        # Parser le fichier ALTO pour extraire les lignes
        tree = ET.parse(alto_path)
        root = tree.getroot()
        ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
        
        # Extraire toutes les lignes
        lines_data = []
        for textline in root.findall('.//alto:TextLine', ns):
            # Obtenir la baseline si disponible
            baseline_str = textline.get('BASELINE', '')
            if baseline_str:
                # Parser la baseline (format: "x1 y1 x2 y2 ...")
                coords = baseline_str.split()
                baseline_points = []
                for i in range(0, len(coords), 2):
                    if i + 1 < len(coords):
                        baseline_points.append([float(coords[i]), float(coords[i+1])])
                
                if len(baseline_points) >= 2:
                    # Obtenir aussi le polygon si disponible
                    polygon = textline.find('.//alto:Polygon', ns)
                    boundary = None
                    if polygon is not None:
                        points_str = polygon.get('POINTS', '')
                        if points_str:
                            # Parser les points du polygon
                            coords = points_str.split()
                            boundary_points = []
                            for i in range(0, len(coords), 2):
                                if i + 1 < len(coords):
                                    boundary_points.append([float(coords[i]), float(coords[i+1])])
                            if boundary_points:
                                boundary = boundary_points
                    
                    # Si pas de boundary, créer une à partir de la baseline
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
            print(f"  Aucune ligne avec baseline trouvée dans {alto_path}")
            return []
        
        # Charger l'image
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Créer les objets BaselineLine pour Kraken
        kraken_lines = []
        for line_data in lines_data:
            line = BaselineLine(
                id=f"line_{len(kraken_lines)}",
                baseline=line_data['baseline'],
                boundary=line_data['boundary'] if line_data.get('boundary') else None
            )
            kraken_lines.append(line)
        
        # Créer l'objet Segmentation
        segmentation = Segmentation(
            type='baselines',
            imagename=image_path,
            text_direction='horizontal-lr',
            script_detection=False,
            lines=kraken_lines,
            regions={}
        )
        
        # Faire la reconnaissance
        predictions = list(rpred.rpred(
            self.model,
            image,
            segmentation,
            pad=16,
            bidi_reordering=True
        ))
        
        # Convertir les résultats
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
            
        # except Exception as e:
        #     print(f"  Erreur HTR: {e}")
        #     # Retourner des résultats vides pour chaque ligne
        #     try:
        #         tree = ET.parse(alto_path)
        #         root = tree.getroot()
        #         ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
        #         results = [{'text': '', 'confidence': 0.0} for _ in range(num_lines)]
        #         num_lines = len(root.findall('.//alto:TextLine', ns))
        #     except:
        #         results = []
        
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
    
    def score(self, pred_path, gt_path):
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
        return 
    
    def predict(self, data_path, output_dir, save_image=True):
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
        
        # Find all ALTO XML files
        alto_files = sorted(glob.glob(os.path.join(data_path, "*.xml")))
        
        if not alto_files:
            raise ValueError(f"No ALTO XML files found in {data_path}")
        
        print(f"Processing {len(alto_files)} ALTO files...")
        
        results = []
        for alto_path in tqdm(alto_files, desc="Recognizing text", unit="page"):
            try:
                # Extract image path from ALTO
                image_path, lines, _ = extract_lines_from_alto(alto_path)
                
                if not os.path.exists(image_path):
                    print(f"Warning: Image {image_path} not found")
                    continue
                
                if not lines:
                    print(f"Warning: No lines found in {alto_path}")
                    continue
                
                # Recognize text - CORRECTION ICI
                # Passer les chemins, pas l'objet image et la liste lines
                recognized_texts = self._recognize_text(image_path, alto_path)
                
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
                    if not os.path.exists(image_output):
                        shutil.copy2(image_path, image_output)
                
            except Exception as e:
                print(f"Error processing {alto_path}: {e}")
                import traceback
                traceback.print_exc()
        
        return results