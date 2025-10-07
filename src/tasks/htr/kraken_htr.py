from src.tasks.base_tasks import BaseTask
import os
import glob
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import shutil
from lxml import etree as ET

import jiwer
from jiwer import cer, wer, mer, wil, wip

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
        

    def _extract_text_from_alto(self, alto_path):
        """
        Extract transcribed text from ALTO XML file.
        
        Args:
            alto_path: Path to ALTO XML file
            
        Returns:
            List of dictionaries with line_id and text content
        """
        tree = ET.parse(alto_path)
        root = tree.getroot()
        ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
        
        lines_text = []
        
        for textline in root.findall('.//alto:TextLine', ns):
            line_id = textline.get('ID', '')
            
            # Extract text from String elements
            strings = textline.findall('.//alto:String', ns)
            if strings:
                # Concatenate all strings in the line with spaces
                text = ' '.join([s.get('CONTENT', '') for s in strings if s.get('CONTENT')])
            else:
                text = ''
            
            lines_text.append({
                'id': line_id,
                'text': text
            })
        
        return lines_text


    def score(self, pred_path, gt_path):
        """
        Compute metrics for the HTR model using jiwer library.
        
        Args:
            pred_path: Path to directory containing prediction ALTO files
            gt_path: Path to directory containing ground truth ALTO files
            
        Returns:
            Dictionary of evaluation metrics
        """
        
        # Find all ground truth files
        gt_files = sorted(glob.glob(os.path.join(gt_path, "*.xml")))
        if not gt_files:
            raise ValueError(f"No ground truth ALTO files found in {gt_path}")
        
        # Collect all ground truth and prediction texts
        all_gt_texts = []
        all_pred_texts = []
        
        total_lines = 0
        perfect_lines = 0
        files_processed = 0
        
        # Process each file
        for gt_file in tqdm(gt_files, desc="Scoring HTR", unit="page"):
            base_name = os.path.basename(gt_file)
            pred_file = os.path.join(pred_path, base_name)
            
            if not os.path.exists(pred_file):
                print(f"Warning: No prediction file found for {base_name}")
                continue
            
            try:
                # Extract text from both files
                gt_lines = self._extract_text_from_alto(gt_file)
                pred_lines = self._extract_text_from_alto(pred_file)
                
                # Match lines by ID
                gt_dict = {line['id']: line['text'] for line in gt_lines}
                pred_dict = {line['id']: line['text'] for line in pred_lines}
                
                # Process matched lines
                for line_id in gt_dict:
                    gt_text = gt_dict[line_id]
                    pred_text = pred_dict.get(line_id, '')
                    
                    # Skip empty ground truth lines
                    if not gt_text.strip():
                        continue
                    
                    total_lines += 1
                    all_gt_texts.append(gt_text)
                    all_pred_texts.append(pred_text)
                    
                    # Count perfect matches
                    if pred_text == gt_text:
                        perfect_lines += 1
                
                files_processed += 1
                
            except Exception as e:
                print(f"Error processing {gt_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not all_gt_texts:
            print("Warning: No valid text found for evaluation")
            return {}
        
        # Calculate metrics using jiwer
        try:
            cer_score = cer(all_gt_texts, all_pred_texts)
            wer_score = wer(all_gt_texts, all_pred_texts)
            mer_score = mer(all_gt_texts, all_pred_texts)
            wil_score = wil(all_gt_texts, all_pred_texts)
            wip_score = wip(all_gt_texts, all_pred_texts)
            
            # Calculate additional metrics
            char_accuracy = 1.0 - cer_score
            word_accuracy = 1.0 - wer_score
            line_accuracy = perfect_lines / total_lines if total_lines > 0 else 0.0
            
            # Get detailed error counts for CER
            cer_output = jiwer.process_characters(all_gt_texts, all_pred_texts)
            
            # Get detailed error counts for WER
            wer_output = jiwer.process_words(all_gt_texts, all_pred_texts)
            
            metrics_dict = {
                "dataset_test/cer": cer_score,
                "dataset_test/wer": wer_score,
                "dataset_test/mer": mer_score,
                "dataset_test/wil": wil_score,
                "dataset_test/wip": wip_score,
                "dataset_test/char_accuracy": char_accuracy,
                "dataset_test/word_accuracy": word_accuracy,
                "dataset_test/line_accuracy": line_accuracy,
                "dataset_test/total_chars": sum(len(text) for text in all_gt_texts),
                "dataset_test/total_words": sum(len(text.split()) for text in all_gt_texts),
                "dataset_test/total_lines": total_lines,
                "dataset_test/perfect_lines": perfect_lines,
                "dataset_test/files_processed": files_processed,
                "dataset_test/char_insertions": cer_output.insertions,
                "dataset_test/char_deletions": cer_output.deletions,
                "dataset_test/char_substitutions": cer_output.substitutions,
                "dataset_test/word_insertions": wer_output.insertions,
                "dataset_test/word_deletions": wer_output.deletions,
                "dataset_test/word_substitutions": wer_output.substitutions,
            }
            
            self._display_metrics(metrics_dict)
            
            return metrics_dict
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
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
                
                # Recognize text
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