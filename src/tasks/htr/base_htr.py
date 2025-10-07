from src.tasks.base_tasks import BaseTask
from abc import abstractmethod
import os
import glob
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from lxml import etree as ET

import jiwer
from jiwer import cer, wer, mer, wil, wip


class BaseHTR(BaseTask):
    """
    Base class for HTR (Handwritten Text Recognition) tasks.
    Provides common functionality for all HTR implementations.
    """
    
    def __init__(self, config):
        """
        Initialize the HTR base class.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
    
    @abstractmethod
    def load(self):
        """Load the model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def predict(self, data_path, output_dir, save_image=True):
        """
        Perform HTR prediction. Must be implemented by subclasses.
        
        Args:
            data_path: Directory containing input data
            output_dir: Directory to save predictions
            save_image: Whether to copy images to output directory
            
        Returns:
            List of prediction results
        """
        pass
    
    def train(self, data_path=None, **kwargs):
        """
        Train the HTR model. Default implementation prints a message.
        Can be overridden by subclasses that support training.
        """
        print(f"Training for {self.name} is not yet implemented.")
    
    def _extract_text_from_alto(self, alto_path):
        """
        Extract transcribed text from ALTO XML file.
        
        Args:
            alto_path: Path to ALTO XML file
            
        Returns:
            Concatenated text from all String elements
        """
        tree = ET.parse(alto_path)
        root = tree.getroot()
        ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
        
        # Extract all String elements
        strings = root.findall('.//alto:String', ns)
        texts = [s.get('CONTENT', '') for s in strings if s.get('CONTENT')]
        
        return ' '.join(texts) if texts else ''
    
    def _extract_lines_text_from_alto(self, alto_path):
        """
        Extract text line by line from ALTO XML file.
        
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
            
            # Extract text from String elements in this line
            strings = textline.findall('.//alto:String', ns)
            if strings:
                text = ' '.join([s.get('CONTENT', '') for s in strings if s.get('CONTENT')])
            else:
                text = ''
            
            lines_text.append({
                'id': line_id,
                'text': text
            })
        
        return lines_text
    
    def _create_simple_alto_with_text(self, image_path, text, output_path):
        """
        Create a simple ALTO XML file with recognized text.
        Creates one TextBlock and one TextLine covering the entire image.
        
        Args:
            image_path: Path to the source image
            text: Recognized text
            output_path: Where to save the ALTO XML
        """
        ns = "http://www.loc.gov/standards/alto/ns-v4#"
        NSMAP = {
            None: ns,
            "xsi": "http://www.w3.org/2001/XMLSchema-instance"
        }
        
        # Get image dimensions
        with Image.open(image_path) as img:
            width, height = img.size
        
        # Create ALTO structure
        alto = ET.Element("alto", nsmap=NSMAP, attrib={
            "{http://www.w3.org/2001/XMLSchema-instance}schemaLocation":
                f"{ns} http://www.loc.gov/standards/alto/v4/alto-4-2.xsd"
        })
        
        # Description
        description = ET.SubElement(alto, "Description")
        ET.SubElement(description, "MeasurementUnit").text = "pixel"
        source_info = ET.SubElement(description, "sourceImageInformation")
        ET.SubElement(source_info, "fileName").text = os.path.basename(image_path)
        
        # Processing
        processing = ET.SubElement(description, "OCRProcessing")
        processing_step = ET.SubElement(processing, "ocrProcessingStep")
        software = ET.SubElement(processing_step, "processingSoftware")
        ET.SubElement(software, "softwareName").text = self.name
        
        # Tags
        tags = ET.SubElement(alto, "Tags")
        ET.SubElement(tags, "OtherTag", ID="BT1", LABEL="MainZone", 
                     DESCRIPTION="block type MainZone")
        ET.SubElement(tags, "OtherTag", ID="LT1", LABEL="DefaultLine",
                     DESCRIPTION="line type DefaultLine")
        
        # Layout
        layout = ET.SubElement(alto, "Layout")
        page = ET.SubElement(layout, "Page", ID="page1", PHYSICAL_IMG_NR="1",
                           HEIGHT=str(height), WIDTH=str(width))
        print_space = ET.SubElement(page, "PrintSpace", 
                                   HEIGHT=str(height), WIDTH=str(width),
                                   VPOS="0", HPOS="0")
        
        # Single TextBlock covering the whole image
        text_block = ET.SubElement(print_space, "TextBlock", ID="block_0",
                                  HPOS="0", VPOS="0",
                                  WIDTH=str(width), HEIGHT=str(height),
                                  TAGREFS="BT1")
        
        # Single TextLine with the recognized text
        margin = 5
        text_line = ET.SubElement(text_block, "TextLine", ID="line_0",
                                HPOS=str(margin), VPOS=str(margin),
                                WIDTH=str(width - 2 * margin), 
                                HEIGHT=str(height - 2 * margin),
                                TAGREFS="LT1")
        
        # Baseline
        baseline_y = height // 2
        text_line.set('BASELINE', f"{margin} {baseline_y} {width - margin} {baseline_y}")
        
        # Shape
        shape = ET.SubElement(text_line, "Shape")
        points = f"{margin} {margin} {width - margin} {margin} {width - margin} {height - margin} {margin} {height - margin}"
        ET.SubElement(shape, "Polygon", POINTS=points)
        
        # String with recognized text
        if text:
            string_elem = ET.SubElement(text_line, "String")
            string_elem.set('CONTENT', text)
            string_elem.set('WC', '1.0')
        
        # Save
        tree = ET.ElementTree(alto)
        tree.write(output_path, pretty_print=True, 
                  xml_declaration=True, encoding="UTF-8")
    
    def score(self, pred_path, gt_path):
        """
        Compute HTR metrics by comparing predictions to ground truth.
        Uses standard metrics: CER, WER, MER, WIL, WIP.
        
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
        
        all_gt_texts = []
        all_pred_texts = []
        
        total_files = 0
        matched_files = 0
        total_lines = 0
        perfect_lines = 0
        
        # Process each file
        for gt_file in tqdm(gt_files, desc="Scoring HTR", unit="page"):
            base_name = os.path.basename(gt_file)
            pred_file = os.path.join(pred_path, base_name)
            
            total_files += 1
            
            if not os.path.exists(pred_file):
                print(f"Warning: No prediction file found for {base_name}")
                continue
            
            try:
                # Try line-by-line comparison first
                gt_lines = self._extract_lines_text_from_alto(gt_file)
                pred_lines = self._extract_lines_text_from_alto(pred_file)
                
                if gt_lines:
                    # Match lines by ID
                    gt_dict = {line['id']: line['text'] for line in gt_lines}
                    pred_dict = {line['id']: line['text'] for line in pred_lines}
                    
                    for line_id in gt_dict:
                        gt_text = gt_dict[line_id]
                        pred_text = pred_dict.get(line_id, '')
                        
                        # Skip empty ground truth lines
                        if not gt_text.strip():
                            continue
                        
                        total_lines += 1
                        all_gt_texts.append(gt_text)
                        all_pred_texts.append(pred_text)
                        
                        if pred_text == gt_text:
                            perfect_lines += 1
                else:
                    # Fallback: compare full text
                    gt_text = self._extract_text_from_alto(gt_file)
                    pred_text = self._extract_text_from_alto(pred_file)
                    
                    if gt_text.strip():
                        all_gt_texts.append(gt_text)
                        all_pred_texts.append(pred_text)
                
                matched_files += 1
                
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
            
            char_accuracy = 1.0 - cer_score
            word_accuracy = 1.0 - wer_score
            line_accuracy = perfect_lines / total_lines if total_lines > 0 else 0.0
            
            # Get detailed error counts
            cer_output = jiwer.process_characters(all_gt_texts, all_pred_texts)
            wer_output = jiwer.process_words(all_gt_texts, all_pred_texts)
            
            metrics_dict = {
                "score/cer": cer_score,
                "score/wer": wer_score,
                "score/mer": mer_score,
                "score/wil": wil_score,
                "score/wip": wip_score,
                "accuracy/char_accuracy": char_accuracy,
                "accuracy/word_accuracy": word_accuracy,
                "accuracy/line_accuracy": line_accuracy,
                "total/total_chars": sum(len(text) for text in all_gt_texts),
                "total/total_words": sum(len(text.split()) for text in all_gt_texts),
                "total/total_lines": total_lines,
                "total/perfect_lines": perfect_lines,
                "total/files_processed": matched_files,
                "detailed/char_insertions": cer_output.insertions,
                "detailed/char_deletions": cer_output.deletions,
                "detailed/char_substitutions": cer_output.substitutions,
                "detailed/word_insertions": wer_output.insertions,
                "detailed/word_deletions": wer_output.deletions,
                "detailed/word_substitutions": wer_output.substitutions,
            }
            
            self._display_metrics(metrics_dict)
            return metrics_dict
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            import traceback
            traceback.print_exc()
            return {}