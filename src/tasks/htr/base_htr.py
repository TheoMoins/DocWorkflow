from src.tasks.base_tasks import BaseTask
from abc import abstractmethod
import os
import glob
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from lxml import etree as ET

from jiwer import cer, wer

from src.utils.metrics import calculate_htr_metrics
from src.utils.alto_text import extract_text_from_alto, extract_lines_text_from_alto

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
    
    def train(self, data_path=None, **kwargs):
        """
        Train the HTR model. Default implementation prints a message.
        Can be overridden by subclasses that support training.
        """
        print(f"Training for {self.name} is not yet implemented.")
    
    
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
        
    
    def _score_batch(self, pred_files, gt_files, pred_dir, gt_dir):
        """
        Score a batch of HTR predictions.
        """
        all_gt_texts = []
        all_pred_texts = []
        page_scores = []
        
        for pred_file, gt_file in tqdm(zip(pred_files, gt_files), total=len(pred_files), desc="  Scoring", unit="page"):
            try:
                gt_lines = extract_lines_text_from_alto(gt_file)
                pred_lines = extract_lines_text_from_alto(pred_file)
                
                page_gt_texts = []
                page_pred_texts = []
                
                if len(pred_lines) == 1 and len(gt_lines) > 1:
                    # Fallback: split prediction by line breaks
                    single_text = pred_lines[0]['text']
                    pred_texts_split = [t.strip() for t in single_text.split('\n') if t.strip()]
                    
                    # If split matches ground truth line count, use it
                    if len(pred_texts_split) == len(gt_lines):
                        for gt_line, pred_text in zip(gt_lines, pred_texts_split):
                            gt_text = gt_line['text']
                            if gt_text.strip():
                                page_gt_texts.append(gt_text)
                                page_pred_texts.append(pred_text)

                    else:
                        full_gt = ' '.join(line['text'] for line in gt_lines if line['text'].strip())
                        full_pred = ' '.join(pred_texts_split)
                        if full_gt.strip():
                            page_gt_texts.append(full_gt)
                            page_pred_texts.append(full_pred)
                else:
                    gt_text = extract_text_from_alto(gt_file)
                    pred_text = extract_text_from_alto(pred_file)
                    if gt_text.strip():
                        page_gt_texts.append(gt_text)
                        page_pred_texts.append(pred_text)
                
                if page_gt_texts and page_pred_texts:
                    all_gt_texts.extend(page_gt_texts)
                    all_pred_texts.extend(page_pred_texts)
                    
                    page_scores.append({
                        'page': Path(gt_file).stem,
                        'cer': cer(page_gt_texts, page_pred_texts),
                        'wer': wer(page_gt_texts, page_pred_texts),
                        'char_count': sum(len(t) for t in page_gt_texts),
                        'word_count': sum(len(t.split()) for t in page_gt_texts)
                    })
            except Exception as e:
                print(f"  Error on {Path(gt_file).name}: {e}")
        
        # Calculate global metrics
        metrics_dict = calculate_htr_metrics(all_gt_texts, all_pred_texts, page_scores)
        
        return metrics_dict, page_scores