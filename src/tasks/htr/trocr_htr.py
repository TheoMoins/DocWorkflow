from src.tasks.htr.base_htr import BaseHTR
import os
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
from pathlib import Path
import shutil
import torch
from lxml import etree as ET

from transformers import TrOCRProcessor, VisionEncoderDecoderModel, GenerationConfig

from IPython.display import display

from src.alto.alto_lines import read_lines_geometry
from src.alto.alto_text import write_text_to_alto

class TrOCRHTRTask(BaseHTR):
    """
    HTR implementation using TrOCR.
    Processes pre-segmented lines from ALTO XML files.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "HTR_TrOCR"
        self.processor = None
        self.model = None
    
    def load(self):
        """
        Load the TrOCR model and processor.
        """
        model_path = self.config.get('model_path', 'medieval-data/trocr-medieval-base')
        
        print(f"Loading TrOCR model: {model_path}")
        
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        
        self.generation_config = GenerationConfig.from_model_config(self.model.config)
        
        # Override with custom settings if needed
        self.generation_config.max_length = self.config.get('max_length', 64)
        self.generation_config.early_stopping = True
        self.generation_config.no_repeat_ngram_size = 3
        self.generation_config.length_penalty = 2.0
        self.generation_config.num_beams = self.config.get('num_beams', 4)
        
        self.to_device()
        self.model.eval()
        
        print(f"TrOCR model loaded on {self.device}")
    
    def _get_file_extensions(self):
        """TrOCR works with ALTO XML files (requires pre-segmented lines)."""
        return ['*.xml']
    
    def _extract_line_image(self, image, boundary):
        """
        Extract a line region from the full page image.
        
        Args:
            image: PIL Image of the full page
            boundary: List of polygon points [[x1,y1], [x2,y2], ...]
            
        Returns:
            PIL Image of the cropped line
        """
        if not boundary:
            return None
        
        # Get bounding box from polygon
        boundary_array = np.array(boundary)
        min_x = int(boundary_array[:, 0].min())
        max_x = int(boundary_array[:, 0].max())
        min_y = int(boundary_array[:, 1].min())
        max_y = int(boundary_array[:, 1].max())

        # Create a new blank mask image (mode 'L' for grayscale 8-bit pixels, 0-255)
        mask_pil = Image.new('L', image.size, 0) # Initialize with black (0)
        # Draw the polygon on the mask with white (255) fill
        draw = ImageDraw.Draw(mask_pil)
        draw.polygon(boundary, fill=255) # have to use boundary, not boundary_array

        # Apply the mask to the original image
        # Image.composite(image1, image2, mask) selects pixels from image1 where mask is white,
        # and from image2 where mask is black. We want to keep original pixels where mask is white,
        # and make others black, so image1 = img_pil, image2 = black image.
        masked_image_pil = Image.composite(image, Image.new('RGB', image.size, (0, 0, 0)), mask_pil)


        # Crop line
        line_image = masked_image_pil.crop((min_x, min_y, max_x, max_y))
        
        return line_image
    
    def _recognize_line(self, line_image):
        """
        Recognize text from a single line image.
        
        Args:
            line_image: PIL Image of the text line
            
        Returns:
            Dictionary with 'text' and 'confidence'
        """
        # Preprocess
        pixel_values = self.processor(
            line_image, 
            return_tensors="pt"
        ).pixel_values.to(self.device)
        
        # Generate using the generation config
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                generation_config=self.generation_config
            )
        
        # Decode
        generated_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        return {
            'text': generated_text,
            'confidence': 1.0
        }
    
    def _process_batch(self, file_paths, source_dir, output_dir, save_image=True, **kwargs):
        """
        Process a batch of ALTO files for TrOCR HTR.
        
        Args:
            file_paths: List of ALTO XML file paths
            source_dir: Source directory
            output_dir: Output directory for ALTO with recognized text
            save_image: Whether to copy images
            
        Returns:
            List of prediction results
        """
        print(f"  Processing {len(file_paths)} ALTO files...")
        
        results = []
        
        for alto_path in tqdm(file_paths, desc="  Recognizing text", unit="page"):
            try:
                image_path, lines, _ = read_lines_geometry(alto_path)
                
                if not os.path.exists(image_path):
                    print(f"  Warning: Image {image_path} not found")
                    continue
                
                tree_check = ET.parse(alto_path)
                ns_check = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
                raw_lines = tree_check.getroot().findall('.//alto:TextLine', ns_check)
                if not raw_lines:
                    print(f"  Warning: No TextLines found in {alto_path}")
                    continue

                is_from_gt = Path(alto_path).parent.resolve() == Path(source_dir).resolve()
                if is_from_gt:
                    print(f"  [GT] Using ground truth line segmentation ({len(raw_lines)} lines) from {os.path.basename(alto_path)}")
                
                # Load page image
                page_image = Image.open(image_path).convert("RGB")
                
                # Recognize each line
                recognized_texts = []
                for line in lines:
                    if not line.get('boundary'):
                        recognized_texts.append({'text': '', 'confidence': 0.0})
                        continue
                    
                    # Extract line image
                    line_image = self._extract_line_image(page_image, line['boundary'])
                    
                    if line_image is None:
                        recognized_texts.append({'text': '', 'confidence': 0.0})
                        continue
                    
                    # Recognize text
                    result = self._recognize_line(line_image)
                    recognized_texts.append(result)
                
                # Create output ALTO
                output_path = os.path.join(output_dir, os.path.basename(alto_path))
                
                if not os.path.exists(output_path):
                    shutil.copy2(alto_path, output_path)
                
                # Add recognized text to ALTO
                write_text_to_alto(output_path, recognized_texts, output_path)
                
                results.append({
                    'file': alto_path,
                    'texts': recognized_texts
                })
                
                # Copy image if requested
                if save_image:
                    image_output = os.path.join(output_dir, os.path.basename(image_path))
                    if not os.path.exists(image_output):
                        shutil.copy2(image_path, image_output)
                
                page_image.close()
                
            except Exception as e:
                print(f"  Error processing {alto_path}: {e}")
                import traceback
                traceback.print_exc()
        
        return results
    