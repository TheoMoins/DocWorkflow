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
            'base_model': config.get('base_model', None),
            'max_pixels': config.get('max_pixels', 512*28*28)
        }
        
        self.processor = None
        self.model = None
        self.is_minicpm = False
        self.tokenizer = None
    
    def _build_base_gen_kwargs(self) -> dict:
        """Resolve EOS/pad token ids and return safe base generation kwargs."""
        gen_kwargs = {}
        eos_id = getattr(self.model.config, "eos_token_id", None)
        pad_id = getattr(self.model.config, "pad_token_id", None)
        tok = getattr(self.processor, "tokenizer", None)
        if eos_id is None and tok is not None:
            eos_id = getattr(tok, "eos_token_id", None)
        if pad_id is None and tok is not None:
            pad_id = getattr(tok, "pad_token_id", None)
        if eos_id is not None:
            gen_kwargs["eos_token_id"] = eos_id
        if pad_id is not None:
            gen_kwargs["pad_token_id"] = pad_id
        return gen_kwargs

    def load(self):
        from peft import PeftModel
        from src.utils.memory_monitor import get_fast_vision_model
        FastVisionModel = get_fast_vision_model()
        if FastVisionModel is None:
            raise RuntimeError("Training requires a GPU with unsloth installed.")

        """Load the VLM model (common for both page and line level)."""
        if not self.model_name:
            raise ValueError("model_name must be specified in config")
        
        print(f"Loading VLM model: {self.model_name}")
        
        base_model_name = self.hyperparams.get('base_model')
        use_lora = base_model_name is not None
        
        base_model = None
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
                    base_model_name, 
                    trust_remote_code=True, 
                    padding_side='left',
                    max_pixels=self.hyperparams['max_pixels']
                )
                base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    base_model_name, **model_kwargs
                )
            elif False: #os.path.exists(base_model_name):
                print("Local checkpoint detected, loading with FastVisionModel (Unsloth)...")
                self.processor = AutoProcessor.from_pretrained(
                    base_model_name, trust_remote_code=True, padding_side='left'
                )
                self.model, _ = FastVisionModel.from_pretrained(
                    base_model_name,
                    load_in_4bit=self.hyperparams.get('use_4bit', False),
                    load_in_8bit=self.hyperparams.get('use_8bit', False),
                )
                FastVisionModel.for_inference(self.model)
            else:
                self.processor = AutoProcessor.from_pretrained(
                    base_model_name, 
                    trust_remote_code=True,
                    padding_side='left',
                    max_pixels=self.hyperparams['max_pixels']
                )
                base_model = AutoModelForImageTextToText.from_pretrained(
                    base_model_name, **model_kwargs
                )

            if base_model is not None:
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
                trust_remote_code=True,
                padding_side='left',
                max_pixels=self.hyperparams['max_pixels'],
            )
            
            # Model loading configuration
            model_kwargs = {"trust_remote_code": True}
            
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
        if not boundary:
            return None

        boundary_array = np.array(boundary)
        min_x = int(boundary_array[:, 0].min())
        max_x = int(boundary_array[:, 0].max())
        min_y = int(boundary_array[:, 1].min())
        max_y = int(boundary_array[:, 1].max())

        w, h = max_x - min_x, max_y - min_y
        if w <= 0 or h <= 0:
            return None

        # Crop d'abord la bbox (petit rectangle)
        cropped = image.crop((min_x, min_y, max_x, max_y))

        # Masque local (taille de la bbox, pas de la page)
        local_boundary = [(int(pt[0]) - min_x, int(pt[1]) - min_y) for pt in boundary]
        mask = Image.new('L', (w, h), 0)
        ImageDraw.Draw(mask).polygon(local_boundary, fill=255)

        # Appliquer le masque
        result = Image.composite(cropped, Image.new('RGB', (w, h), (0, 0, 0)), mask)

        max_ratio = 190 # unsloth limit: 200
        rw, rh = result.size
        if rw > 0 and rh > 0:
            ratio = max(rw / rh, rh / rw)
            if ratio > max_ratio:
                if rw > rh:
                    new_h = max(1, rw // max_ratio)
                    padded = Image.new('RGB', (rw, new_h), (0, 0, 0))
                    padded.paste(result, (0, (new_h - rh) // 2))
                    result = padded
                else:
                    new_w = max(1, rh // max_ratio)
                    padded = Image.new('RGB', (new_w, rh), (0, 0, 0))
                    padded.paste(result, ((new_w - rw) // 2, 0))
                    result = padded

        return result


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
            enable_thinking=False,
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
        gen_kwargs = self._build_base_gen_kwargs()
        gen_kwargs.update({"max_new_tokens": self.max_new_tokens, "do_sample": False})
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
        
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
