from src.tasks.htr.base_htr import BaseHTR
from abc import abstractmethod
import torch
import numpy as np
from PIL import Image, ImageDraw
import glob
import os

from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_vl_utils import process_vision_info
from src.utils.transformers_models import is_supported_by_auto_image_text


class BaseVLMHTR(BaseHTR):
    """
    Base class for VLM-based HTR tasks.
    Handles model loading and common VLM operations.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        self.model_name = config.get('model_name')
        self.max_new_tokens = config.get('max_new_tokens', 512)
        self.prompt = config.get('prompt', 
                                 "Transcribe the text from this image.")
        
        # Hyperparameters for model loading
        self.hyperparams = {
            'use_dtype_param': config.get('use_dtype_param', False),
            'device_map': config.get('device_map'),
            'attn_implementation': config.get('attn_implementation'),
            'model_class': config.get('model_class', None),
            'use_4bit': config.get('use_4bit', False),
            'use_8bit': config.get('use_8bit', False),
            'base_model': config.get('base_model', None)
        }
        
        self.processor = None
        self.model = None
        self.is_minicpm = False
        self.tokenizer = None
    
    def load(self):
        from peft import PeftModel

        """Load the VLM model (common for both page and line level)."""
        if not self.model_name:
            raise ValueError("model_name must be specified in config")
        
        print(f"Loading VLM model: {self.model_name}")
        
        base_model_name = self.hyperparams.get('base_model')
        use_lora = base_model_name is not None
        
        if use_lora:
            # LoRA adapter mode: load base model + adapter
            print(f"Loading base model: {base_model_name}")
            print(f"Loading LoRA adapter from: {self.model_name}")

            # Model loading configuration
            model_kwargs = {"trust_remote_code": True}
            
            if self.device == 'cuda':
                model_kwargs['dtype'] = torch.bfloat16
            else:
                model_kwargs['dtype'] = torch.float32
            
            if self.hyperparams['device_map']:
                model_kwargs['device_map'] = self.hyperparams['device_map']
            else:
                model_kwargs['device_map'] = 'auto'

            model_class_name = self.hyperparams.get('model_class')
            base_model_lower = base_model_name.lower()

            if model_class_name == 'MiniCPM' or 'minicpm' in base_model_lower:
                print("Using MiniCPM base model with LoRA adapter")

                _resampler_files = glob.glob(
                    os.path.expanduser("/mnt/theo/cache/huggingface/modules/transformers_modules/**/resampler.py"),
                    recursive=True
                )
                for _f in _resampler_files:
                    with open(_f, 'r') as _fh:
                        _content = _fh.read()
                    if 'from typing import' in _content and 'List' not in _content.split('from typing import')[1].split('\n')[0]:
                        _content = _content.replace('from typing import ', 'from typing import List, ', 1)
                        with open(_f, 'w') as _fh:
                            _fh.write(_content)
                        print(f"Patched typing imports in {_f}")
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    base_model_name, trust_remote_code=True
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    trust_remote_code=True,
                    attn_implementation='sdpa',
                    torch_dtype=torch.bfloat16,
                    device_map='auto' if self.device == 'cuda' else None
                )
                self.is_minicpm = True
            elif 'qwen' in base_model_lower and 'vl' in base_model_lower:
                print("Using Qwen3VLForConditionalGeneration for base model")
                from transformers import Qwen3VLForConditionalGeneration
                self.processor = AutoProcessor.from_pretrained(
                    base_model_name, trust_remote_code=True, padding_side='left'
                )
                base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    base_model_name, **model_kwargs
                )
            else:
                self.processor = AutoProcessor.from_pretrained(
                    base_model_name, trust_remote_code=True
                )
                base_model = AutoModelForImageTextToText.from_pretrained(
                    base_model_name, **model_kwargs
                )
            
            # Load LoRA adapter
            print("Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(
                base_model,
                self.model_name,
                is_trainable=False
            )
            
        else:
            # Standard loading (no LoRA)
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Model loading configuration
            model_kwargs = {"trust_remote_code": True, 
                            "padding_side": 'left'}
            
            if self.device == 'cuda':
                if self.hyperparams['use_dtype_param']:
                    model_kwargs['dtype'] = torch.bfloat16
                else:
                    model_kwargs['dtype'] = torch.bfloat16
            else:
                if self.hyperparams['use_dtype_param']:
                    model_kwargs['dtype'] = torch.float32
                else:
                    model_kwargs['dtype'] = torch.float32
            
            if self.hyperparams['device_map']:
                model_kwargs['device_map'] = self.hyperparams['device_map']
            
            if self.hyperparams['attn_implementation']:
                model_kwargs['attn_implementation'] = self.hyperparams['attn_implementation']
            
            # Determine model class
            model_class_name = self.hyperparams.get('model_class')
            model_name_lower = self.model_name.lower()
            
            # Try to load the appropriate model class
            if model_class_name == 'Qwen3VL' or ('qwen' in model_name_lower and 'vl' in model_name_lower):
                print("Using Qwen3VLForConditionalGeneration")
                from transformers import Qwen3VLForConditionalGeneration
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_name, **model_kwargs
                )
            elif model_class_name == 'AutoModel':
                self.model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
            elif model_class_name == 'AutoModelForImageTextToText':
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name, **model_kwargs
                )
            elif model_class_name == 'MiniCPM' or (base_model_name and 'minicpm' in base_model_name.lower()):                
                print("Using MiniCPM base model with LoRA adapter")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    base_model_name, trust_remote_code=True
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name, **model_kwargs
                )
                self.model = PeftModel.from_pretrained(
                    base_model, self.model_name, is_trainable=False
                )
                self.is_minicpm = True
            else:
                # Auto-detection
                if is_supported_by_auto_image_text(self.model_name):
                    print("Using AutoModelForImageTextToText (auto-detected)")
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        self.model_name, **model_kwargs
                    )
                else:
                    print("Using AutoModel (fallback)")
                    self.model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
        
        # Move to device if not using device_map=auto
        if not self.hyperparams.get('device_map'):
            self.model.to(self.device)
        
        self.model.eval()
        
        print(f"Model loaded successfully")
            
    def _extract_line_image(self, image, boundary):
        """
        Extract a line region from the full page image.
        
        Args:
            image: PIL Image of the full page
            boundary: List of polygon points [[x1,y1], [x2,y2], ...]
            
        Returns:
            PIL Image of the cropped line or None
        """
        if not boundary:
            return None
        
        # Get bounding box from polygon
        boundary_array = np.array(boundary)
        min_x = int(boundary_array[:, 0].min())
        max_x = int(boundary_array[:, 0].max())
        min_y = int(boundary_array[:, 1].min())
        max_y = int(boundary_array[:, 1].max())

        # Some basic image manipulation code & comments generated by Gemini 2.5 Flash
        # Create a new blank mask image (mode 'L' for grayscale 8-bit pixels, 0-255)
        mask_pil = Image.new('L', image.size, 0) # Initialize with black (0)

        # Draw the polygon on the mask with white (255) fill
        draw = ImageDraw.Draw(mask_pil)
        draw.polygon(boundary, fill=255)

        # Apply the mask to the original image
        # Image.composite(image1, image2, mask) selects pixels from image1 where mask is white,
        # and from image2 where mask is black. We want to keep original pixels where mask is white,
        # and make others black, so image1 = img_pil, image2 = black image.
        masked_image_pil = Image.composite(image, Image.new('RGB', image.size, (0, 0, 0)), mask_pil)

        
        # Add small margin
        #margin = 5
        #min_x = max(0, min_x - margin)
        #min_y = max(0, min_y - margin)
        #max_x = min(image.width, max_x + margin)
        #max_y = min(image.height, max_y + margin)
        
        # Crop line
        line_image = masked_image_pil.crop((min_x, min_y, max_x, max_y))
        
        return line_image
    
    def _compress_image_if_needed(self, image: Image.Image, max_bytes: int = 0.1 * 1024 * 1024) -> Image.Image:
        """
        Compress/resize image if its estimated file size exceeds max_bytes.
        Progressively reduces quality then dimensions until under threshold.
        """
        import io

        def get_jpeg_size(img, quality=85):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            return buf.tell(), buf

        size, buf = get_jpeg_size(image)
        if size <= max_bytes:
            return image

        # Step 1: try reducing JPEG quality
        for quality in [75, 60, 45]:
            size, buf = get_jpeg_size(image, quality)
            if size <= max_bytes:
                print(f"  Image compressed to quality={quality} ({size / 1024 / 1024:.1f} MB)")
                buf.seek(0)
                return Image.open(buf).convert("RGB")

        # Step 2: reduce dimensions
        scale = 0.8
        img = image.copy()
        while scale > 0.1:
            new_w = int(img.width * scale)
            new_h = int(img.height * scale)
            img_resized = img.resize((new_w, new_h), Image.LANCZOS)
            size, buf = get_jpeg_size(img_resized)
            if size <= max_bytes:
                print(f"  Image resized to {new_w}x{new_h} ({size / 1024 / 1024:.1f} MB)")
                buf.seek(0)
                return Image.open(buf).convert("RGB")
            scale -= 0.1

        print(f"  Warning: could not compress image below {max_bytes / 1024 / 1024:.0f} MB")
        return image


    def _prepare_messages(self, image, prompt=None):
        """
        Prepare messages for VLM input.
        
        Args:
            image: PIL Image
            prompt: Text prompt (uses self.prompt if None)
            
        Returns:
            Messages list for the model
        """
        if prompt is None:
            prompt = self.prompt
        
        image = self._compress_image_if_needed(image)

        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    
    #TODO: is messages supposed to be plural here?
    def _generate_from_messages(self, messages):
        """
        Generate text from messages (common generation logic).
        
        Args:
            messages: Messages list
            
        Returns:
            Generated text string
        """
        #TODO: what's going on with minicpm??
        if self.is_minicpm:
            image = messages[0]['content'][0]['image']
            prompt = messages[0]['content'][1]['text']
            minicpm_msgs = [{"role": "user", "content": [prompt, image]}]
            with torch.no_grad():
                result = self.model.chat(
                    image=image,
                    msgs=minicpm_msgs,
                    tokenizer=self.tokenizer,
                    max_new_tokens=self.max_new_tokens
                )
            return result.strip()

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process vision info
        try:
            image_inputs, _ = process_vision_info(messages)
        except:
            # Fallback for models that don't need it
            image_inputs = [messages[0]['content'][0]['image']]
        
        # Prepare inputs
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
                max_new_tokens=self.max_new_tokens,
                do_sample=False
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
        )[0]
        
        # Cleanup
        del inputs, generated_ids, generated_ids_trimmed
        
        return output_text.strip()
    
    @abstractmethod
    def _process_batch(self, file_paths, source_dir, output_dir, save_image=True, **kwargs):
        """Must be implemented by subclasses (page vs line level)."""
        pass