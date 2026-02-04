from src.tasks.htr.base_htr import BaseHTR
from abc import abstractmethod
import torch
import numpy as np
from PIL import Image

from peft import PeftModel
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModel, Qwen3VLForConditionalGeneration
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
        }
        
        self.processor = None
        self.model = None
    
    def load(self):
        """Load the VLM model (common for both page and line level)."""
        if not self.model_name:
            raise ValueError("model_name must be specified in config")
        
        print(f"Loading VLM model: {self.model_name}")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Model loading configuration
        model_kwargs = {"trust_remote_code": True}
        
        if self.device == 'cuda':
            if self.hyperparams['use_dtype_param']:
                model_kwargs['dtype'] = torch.bfloat16
            else:
                model_kwargs['torch_dtype'] = torch.bfloat16
        else:
            if self.hyperparams['use_dtype_param']:
                model_kwargs['dtype'] = torch.float32
            else:
                model_kwargs['torch_dtype'] = torch.float32
        
        if self.hyperparams['device_map']:
            model_kwargs['device_map'] = self.hyperparams['device_map']
        
        if self.hyperparams['attn_implementation']:
            model_kwargs['attn_implementation'] = self.hyperparams['attn_implementation']
        
        # Determine which Auto class to use
        model_class_name = self.hyperparams.get('model_class')
        base_model_name = self.hyperparams.get('base_model')

        # Special handling for explicit model class
        if base_model_name is not None and "Qwen" in base_model_name:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                base_model_name,
                **model_kwargs
            )
            self.model = PeftModel.from_pretrained(self.model, self.model_name)

        if model_class_name == 'AutoModel':
            self.model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
        elif model_class_name == 'AutoModelForImageTextToText':
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name, **model_kwargs
            )
        else:
            if is_supported_by_auto_image_text(self.model_name):
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name, **model_kwargs
                )
            else:
                self.model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
    
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
        
        boundary_array = np.array(boundary)
        min_x = int(boundary_array[:, 0].min())
        max_x = int(boundary_array[:, 0].max())
        min_y = int(boundary_array[:, 1].min())
        max_y = int(boundary_array[:, 1].max())
        
        # Add margin
        margin = 5
        min_x = max(0, min_x - margin)
        min_y = max(0, min_y - margin)
        max_x = min(image.width, max_x + margin)
        max_y = min(image.height, max_y + margin)
        
        return image.crop((min_x, min_y, max_x, max_y))
    
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
        
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    
    def _generate_from_messages(self, messages):
        """
        Generate text from messages (common generation logic).
        
        Args:
            messages: Messages list
            
        Returns:
            Generated text string
        """
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
        )[0]
        
        # Cleanup
        del inputs, generated_ids, generated_ids_trimmed
        
        return output_text.strip()
    
    @abstractmethod
    def _process_batch(self, file_paths, source_dir, output_dir, save_image=True, **kwargs):
        """Must be implemented by subclasses (page vs line level)."""
        pass