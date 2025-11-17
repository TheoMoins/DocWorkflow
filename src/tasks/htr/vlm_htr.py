from src.tasks.htr.base_htr import BaseHTR

import os
import glob
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import shutil
import torch
import gc

from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModel
from src.utils.transformers_models import is_supported_by_auto_image_text
from qwen_vl_utils import process_vision_info

class VLMHTRTask(BaseHTR):
    """
    HTR implementation using CHURRO VLM.
    Processes images directly without requiring line segmentation.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Model config
        self.model_name = config.get('model_name', 'stanford-oval/churro-3B')
        self.name = "HTR_" + self.model_name.split('/')[-1]
        self.max_new_tokens = config.get('max_new_tokens', 512)
        self.batch_size = config.get('batch_size', 1)

        # Prompt template
        self.prompt = config.get(
            'prompt',
            "You are an expert in transcription of historical documents from various languages. "
            "Your task is to extract the full text from a given page."
            "Use exactly ONE line break between each line of text."
        )

        self.processor = None
        self.model = None
    
    def load(self):
        """
        Load the VLM model from Transformers.
        """
        
        print(f"Loading VLM model: {self.model_name}")
        print("This may take several minutes on first run...")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Configuration options for model loading
        model_kwargs = {
            "trust_remote_code": True,
        }
        
        # Handle torch_dtype vs dtype deprecation
        if self.device == 'cuda':
            if self.config.get('use_dtype_param', False):
                model_kwargs['dtype'] = torch.bfloat16
            else:
                model_kwargs['torch_dtype'] = torch.bfloat16
        else:
            if self.config.get('use_dtype_param', False):
                model_kwargs['dtype'] = torch.float32
            else:
                model_kwargs['torch_dtype'] = torch.float32
        
        # Add optional config parameters
        if self.config.get('device_map'):
            model_kwargs['device_map'] = self.config['device_map']
        
        if self.config.get('attn_implementation'):
            model_kwargs['attn_implementation'] = self.config['attn_implementation']
        
        # Determine which Auto class to use
        model_class_name = self.config.get('model_class', None)
        
        if model_class_name == 'AutoModel':
            # Explicitement demandé d'utiliser AutoModel
            print("Using AutoModel (explicitly specified)")
            self.model = AutoModel.from_pretrained(
                self.model_name,
                **model_kwargs
            )
        elif model_class_name == 'AutoModelForImageTextToText':
            # Explicitement demandé d'utiliser AutoModelForImageTextToText
            print("Using AutoModelForImageTextToText (explicitly specified)")
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                **model_kwargs
            )
        else:
            # Auto-détection
            if is_supported_by_auto_image_text(self.model_name):
                print("Using AutoModelForImageTextToText (auto-detected)")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            else:
                print("Model not supported by AutoModelForImageTextToText, using AutoModel")
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
        
        # Move to device if not using device_map
        if not self.config.get('device_map'):
            self.model.to(self.device)
        
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    def _recognize_single_image(self, image_path):
        """
        Recognize text from a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Recognized text as string
        """
        
        # Load and prepare image
        image = Image.open(image_path).convert("RGB")
        
        # Prepare messages for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        
        # Process inputs
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        try:
            image_inputs, _ = process_vision_info(messages)
        except (ImportError, Exception):
            # Fallback for models that don't need it
            image_inputs = [image]
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Generate
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens
        }
        
        # Add optional generation parameters
        if self.config.get('temperature'):
            generation_kwargs['temperature'] = self.config['temperature']
        if self.config.get('top_p'):
            generation_kwargs['top_p'] = self.config['top_p']
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                **generation_kwargs
            )

        # Decode
        generated_ids_trimmed = [
            output[len(input_ids):]
            for input_ids, output in zip(inputs["input_ids"], generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        # Cleanup
        image.close()
        del inputs, generated_ids, generated_ids_trimmed
        
        return output_text[0]

    def _split_churro_output_into_lines(self, alto_path, text, image_path):
        """
        Split CHURRO's single-line output into multiple TextLines based on line breaks.
        Tries to match with existing layout structure if available.
        
        Args:
            alto_path: Path to the ALTO file (may have layout info)
            text: Full text from CHURRO
            image_path: Path to source image
            
        Returns:
            Path to updated ALTO file
        """
        from lxml import etree as ET
        from PIL import Image
        
        # Split text by line breaks
        lines_text = [line.strip() for line in text.split('\n') if line.strip()]
        
        if not lines_text:
            return alto_path
        
        # Try to load existing ALTO structure
        tree = ET.parse(alto_path)
        root = tree.getroot()
        ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
        
        # Get image dimensions
        with Image.open(image_path) as img:
            width, height = img.size
        
        # Find existing TextLines (from layout/line segmentation)
        existing_lines = root.findall('.//alto:TextLine', ns)
        
        if existing_lines and len(existing_lines) == len(lines_text):
            # Perfect match: update existing lines with CHURRO text
            for line_elem, line_text in zip(existing_lines, lines_text):
                # Remove old String elements
                for string_elem in line_elem.findall('alto:String', ns):
                    line_elem.remove(string_elem)
                
                # Add new String with CHURRO text
                string_elem = ET.SubElement(line_elem, f"{{{ns['alto']}}}String")
                string_elem.set('CONTENT', line_text)
                string_elem.set('WC', '1.0')
        
        else:
            # No match: create new structure with estimated line positions
            text_blocks = root.findall('.//alto:TextBlock', ns)
            
            if text_blocks:
                # Use first text block
                text_block = text_blocks[0]
                
                # Remove existing TextLines
                for line_elem in text_block.findall('alto:TextLine', ns):
                    text_block.remove(line_elem)
                
                # Create new TextLines with estimated positions
                line_height = height // max(len(lines_text), 1)
                margin = 10
                
                for idx, line_text in enumerate(lines_text):
                    y_pos = idx * line_height + margin
                    line_elem = ET.SubElement(text_block, f"{{{ns['alto']}}}TextLine")
                    line_elem.set('ID', f'line_{idx}')
                    line_elem.set('HPOS', str(margin))
                    line_elem.set('VPOS', str(y_pos))
                    line_elem.set('WIDTH', str(width - 2 * margin))
                    line_elem.set('HEIGHT', str(line_height - margin))
                    
                    baseline_y = y_pos + line_height // 2
                    line_elem.set('BASELINE', f"{margin} {baseline_y} {width - margin} {baseline_y}")
                    
                    # Add String
                    string_elem = ET.SubElement(line_elem, f"{{{ns['alto']}}}String")
                    string_elem.set('CONTENT', line_text)
                    string_elem.set('WC', '1.0')
        
        # Save updated ALTO
        tree.write(alto_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")
        return alto_path


    def predict(self, data_path, output_dir, save_image=True):
        """
        Run CHURRO HTR on images.
        
        Args:
            data_path: Directory containing images
            output_dir: Directory to save ALTO XML files
            save_image: Whether to copy images to output directory
            
        Returns:
            List of results
        """
        if not self.model:
            self.load()
        
        # Find all images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(data_path, ext)))
        
        if not image_paths:
            raise ValueError(f"No images found in {data_path}")
        
        print(f"Found {len(image_paths)} images")
        
        results = []
        
        # Process images
        for image_path in tqdm(image_paths, desc="Recognizing text", unit="image"):
            try:
                # Recognize text
                text = self._recognize_single_image(image_path)
                
                # Create basic ALTO file
                base_name = Path(image_path).stem
                output_path = os.path.join(output_dir, f"{base_name}.xml")
                
                # Check if there's an existing ALTO with layout/lines
                existing_alto = os.path.join(data_path, f"{base_name}.xml")
                if os.path.exists(existing_alto):
                    # Copy existing structure
                    shutil.copy2(existing_alto, output_path)
                else:
                    # Create simple ALTO
                    self._create_simple_alto_with_text(image_path, text, output_path)
                
                # Split CHURRO output into lines
                self._split_churro_output_into_lines(output_path, text, image_path)
                
                results.append({
                    'file': image_path,
                    'text': text
                })
                
                # Copy image if requested
                if save_image:
                    image_output = os.path.join(output_dir, os.path.basename(image_path))
                    if not os.path.exists(image_output):
                        shutil.copy2(image_path, image_output)
                
                # Clean up memory periodically
                if len(results) % 10 == 0:
                    gc.collect()
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return results