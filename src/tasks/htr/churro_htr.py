from src.tasks.htr.base_htr import BaseHTR

import os
import glob
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import shutil
import torch
import gc

from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

class ChurroHTRTask(BaseHTR):
    """
    HTR implementation using CHURRO VLM.
    Processes images directly without requiring line segmentation.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "HTR (CHURRO)"
        
        # Model config
        self.model_name = config.get('model_name', 'stanford-oval/churro-3B')
        self.max_new_tokens = config.get('max_new_tokens', 512)
        self.batch_size = config.get('batch_size', 1)

        # Prompt template
        self.prompt = config.get(
            'prompt',
            "You are an expert in transcription of historical documents from various languages. "
            "Your task is to extract the full text from a given page in Markdown format."
        )

        self.processor = None
        self.model = None
    
    def load(self):
        """
        Load the CHURRO model.
        """
        
        print(f"Loading CHURRO model: {self.model_name}")
        print("This may take several minutes on first run...")
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32
        )
        
        # Move to device
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
        
        image_inputs, _ = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens
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
                
                # Create ALTO file
                base_name = Path(image_path).stem
                output_path = os.path.join(output_dir, f"{base_name}.xml")
                self._create_simple_alto_with_text(image_path, text, output_path)
                
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