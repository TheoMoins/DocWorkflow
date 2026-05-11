from src.tasks.htr.base_htr import BaseHTR
from abc import abstractmethod
import torch
import numpy as np
from PIL import Image, ImageDraw
import glob
import os
import gc
from pathlib import Path
from src.tasks.htr.prompt_convention import build_conventions_block, load_conventions
from src.content.weighted_sampling import special_char_density

from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_vl_utils import process_vision_info
from src.utils.transformers_models import is_supported_by_auto_image_text
from src.utils.lazy_dataset import LazyLineDataset

from transformers import TrainerCallback

    
class CEREvalCallback(TrainerCallback):
    """Calcule le CER sur le validation set à chaque eval."""

    def __init__(self, model, processor, eval_samples, device):
        self.eval_samples = eval_samples
        self.processor = processor
        self.device = device
        self._model = model

    def on_evaluate(self, args, state, control, **kwargs):
        from jiwer import cer
        model = kwargs.get("model", self._model)
        model.eval()

        gt_texts, pred_texts = [], []

        for sample in self.eval_samples:
            try:
                msg = sample["messages"]
                img_content = next(c for c in msg[0]["content"] if c["type"] == "image")
                img = img_content["image"]
                gt = next(c["text"] for c in msg[1]["content"] if c["type"] == "text")

                inputs = self.processor.apply_chat_template(
                    [msg[:1]],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    enable_thinking=False,
                ).to(self.device)

                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,          # greedy pour le CER
                        enable_thinking= False
                    )
                trimmed = out[0][inputs["input_ids"].shape[1]:]
                pred = self.processor.decode(trimmed, skip_special_tokens=True).strip()

                gt_texts.append(gt)
                pred_texts.append(pred)

                del inputs, out, trimmed
                torch.cuda.empty_cache()

            except Exception:
                continue

        if gt_texts:
            cer_score = cer(gt_texts, pred_texts)
            print(f"\n📊 Eval CER (n={len(gt_texts)}): {cer_score:.4f}")

            print(f"\n📝 Exemples de transcriptions (step {state.global_step}):")
            for i in range(3):
                print(f"  [{i+1}] GT  : {gt_texts[i]}")
                print(f"       Pred: {pred_texts[i]}")

            if args.report_to and "wandb" in args.report_to:
                import wandb
                wandb.log({"eval/cer": cer_score}, step=state.global_step)

        torch.cuda.empty_cache()



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
        self.batch_size = config.get('line_batch_size', 1)
        self.train_sources = config.get('train_sources', None)
        self.prompt_template = config.get('prompt_template', self.prompt)

        # Hyperparameters for model loading
        self.hyperparams = {
            'use_dtype_param': config.get('use_dtype_param', False),
            'device_map': config.get('device_map'),
            'attn_implementation': config.get('attn_implementation'),
            'model_class': config.get('model_class', None),
            'use_4bit': config.get('use_4bit', False),
            'use_8bit': config.get('use_8bit', False),
            'base_model': config.get('base_model', None),
            'max_pixels': config.get('max_pixels', 512*28*28),
            'lora_r': config.get('lora_r', 16),
            'lora_dropout': config.get('lora_dropout', 0),
            'use_rslora': config.get('use_rslora', False),
            'max_seq_length': config.get('max_seq_length', 1024),
            'model_dir': config.get('model_dir', 'src/tasks/htr/models'),
            'train_batch_size': config.get('train_batch_size', 4),
            'gradient_accumulation_steps': config.get('gradient_accumulation_steps', 4),
            'warmup_ratio': config.get('warmup_ratio', 0.1),
            'epochs': config.get('epochs', 3),
            'learning_rate': config.get('learning_rate', 2e-4),
            'weight_decay': config.get('weight_decay', 0.01),
            'dataset_num_proc': config.get('dataset_num_proc', 1),
            'special_char_weighting': config.get('special_char_weighting', None),
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
        """Load the VLM model (common for both page and line level)."""
        from peft import PeftModel
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
    
    def _prepare_samples_with_conventions(self, data_path: Path, prompt_tpl: str) -> list:
        data_path = Path(data_path)
        subdirs = sorted([p for p in data_path.iterdir() if p.is_dir()]) if data_path.exists() else []
        samples = []

        if subdirs:
            docs_with_conv = 0
            for doc_path in subdirs:
                conventions = load_conventions(doc_path)
                if conventions and '{conventions}' in prompt_tpl:
                    resolved_prompt = prompt_tpl.replace('{conventions}', build_conventions_block(conventions))
                    docs_with_conv += 1
                else:
                    resolved_prompt = prompt_tpl.replace('{conventions}', '').strip()
                doc_samples = self._prepare_training_data(doc_path)
                for s in doc_samples:
                    s['prompt'] = resolved_prompt
                samples.extend(doc_samples)
            print(f"  {data_path}: {len(subdirs)} documents, {len(samples)} samples, {docs_with_conv}/{len(subdirs)} with conventions")
        else:
            conventions = load_conventions(data_path)
            if conventions and '{conventions}' in prompt_tpl:
                resolved_prompt = prompt_tpl.replace('{conventions}', build_conventions_block(conventions))
            else:
                resolved_prompt = prompt_tpl.replace('{conventions}', '').strip()
            samples = self._prepare_training_data(data_path)
            for s in samples:
                s['prompt'] = resolved_prompt
            conv_info = f"conventions: {list(conventions.keys())}" if conventions else "no conventions"
            print(f"  {data_path}: {len(samples)} samples ({conv_info})")

        return samples


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
    
    def _convert_set(self, examples):
        return LazyLineDataset(examples, self._format_conversation)
    
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
        gen_kwargs.update({"max_new_tokens": self.max_new_tokens, 
                           "do_sample": False})#,
                           #"enable_thinking": False})
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
    
    
    def _create_finetuned_config(self, output_dir, global_path, task_type: str):
        """
        Create a configuration file for the fine-tuned model.

        Args:
            output_dir: Directory where the model was saved
            global_path: Root data path (parent of train/valid/test)
            task_type: Value for tasks.htr.type in the YAML config

        Returns:
            Path to the created config file
        """
        import yaml
        from pathlib import Path

        config = {
            'run_name': f"{self.name}_finetuned",
            'output_dir': 'results',
            'device': self.config.get('device', 'cuda'),
            'use_wandb': self.config.get('use_wandb', False),
            'wandb_project': self.config.get('wandb_project', 'HTR-comparison'),
            'data': {
                'train': global_path + '/train',
                'valid': global_path + '/valid',
                'test': global_path + '/test',
            },
            'tasks': {
                'htr': {
                    'type': task_type,
                    'config': {
                        'model_name': output_dir,
                        'base_model': self.model_name,
                        'use_lora_adapter': True,
                        'max_new_tokens': self.max_new_tokens,
                        'batch_size': self.batch_size,
                        'prompt': self.prompt,
                        **{k: v for k, v in self.hyperparams.items()
                        if k in ['use_dtype_param', 'device_map', 'attn_implementation', 'model_class']
                        and v is not None},
                    },
                }
            },
        }

        model_config_path = Path(output_dir) / 'inference_config.yml'
        with open(model_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"\n Inference config saved to: {model_config_path}")
        return model_config_path

    def train(self, data_path=None, seed=42):
        from unsloth import FastVisionModel
        from unsloth.trainer import UnslothVisionDataCollator
        from transformers import AutoProcessor
        from trl import SFTTrainer, SFTConfig
        """
        Fine-tune the VLM model at line level using Unsloth.
        Each training sample is a cropped TextLine image + its ground truth text.
        """
        print("To train this model, you must change the environment to vlm-training:")
        print("\n  source envs/vlm-training/bin/activate")

        if not data_path:
            raise ValueError("Training data path is required")

        data_paths = data_path if isinstance(data_path, list) else [data_path]

        global_path = str(data_paths[0].parent)

        print(f"Starting VLM line-level fine-tuning with Unsloth")
        print(f"Model: {self.model_name}")

        print("Preparing line-level training data...")
        prompt_tpl = self.config.get('prompt_template', self.prompt)

        train_samples = []
        for src_path in data_paths:
            train_samples.extend(self._prepare_samples_with_conventions(src_path, prompt_tpl))

        valid_samples = self._prepare_samples_with_conventions(Path(global_path) / "valid", prompt_tpl)

        if not train_samples:
            raise ValueError("No valid line-level training samples found")

        print(f"Found {len(train_samples)} line samples (train) and {len(valid_samples)} (valid)")

        print("Validating train samples...")
        valid_train = self._validate_samples(train_samples)
        print("Validated")
        if len(valid_train) < len(train_samples):
            print(f"  Skipped {len(train_samples) - len(valid_train)} samples")

        # --- Calcul des poids ---
        # alpha contrôle le boost des lignes avec caractères spéciaux
        # alpha=10 → une ligne à densité 0.1 a un poids 2x plus élevé qu'une ligne vide
        # alpha=50 → une ligne à densité 0.1 a un poids 6x plus élevé
        alpha = self.hyperparams.get('special_char_weighting', 0)
        if alpha != 0:
            densities = [special_char_density(s["text"]) for s in valid_train]
            sample_weights = [1.0 + alpha * d for d in densities]

            # Statistiques pour diagnostic
            n_with_special = sum(1 for d in densities if d > 0)
            avg_density = sum(densities) / len(densities) if densities else 0
            print(f"\n📊 Density of special characters:")
            print(f"  Lines with ≥1 special char : {n_with_special} / {len(valid_train)} ({100*n_with_special/len(valid_train):.1f}%)")
            print(f"  Mean density             : {avg_density:.4f}")
            print(f"  Weight min / max             : {min(sample_weights):.2f} / {max(sample_weights):.2f}")
            
            # Resampling
            rng = torch.Generator()
            rng.manual_seed(seed)
            weighted_indices = torch.multinomial(
                torch.tensor(sample_weights, dtype=torch.float32),
                num_samples=len(valid_train),
                replacement=True,
                generator=rng,
            ).tolist()

            valid_train_weighted = [valid_train[i] for i in weighted_indices]
            print(f"✓ Density-weighted resampling: {len(valid_train_weighted)} samples")

            # Verification
            new_avg_density = sum(densities[i] for i in weighted_indices) / len(weighted_indices)
            print(f"  Average density after resampling: {new_avg_density:.4f}")
            n_special_after = sum(1 for i in weighted_indices if densities[i] > 0)
            print(f"  Lines with ≥1 special char after resampling : {n_special_after} ({100*n_special_after/len(weighted_indices):.1f}%)")
        else:
            valid_train_weighted = valid_train

        converted_train_set = self._convert_set(valid_train_weighted)

        if valid_samples:
            print('Validating validation samples')
            valid_valid = valid_train = self._validate_samples(valid_samples)
            print('Validated')
            converted_valid_set = self._convert_set(valid_valid)
        else:
            converted_valid_set = None

        print("Loading model with Unsloth...")
        base_model = self.hyperparams.get('base_model')
        if base_model:
           # model, tokenizer = FastVisionModel.from_pretrained(
           #     base_model,
           #     load_in_4bit=False,
           #     load_in_8bit=False,
           #     use_gradient_checkpointing="unsloth",
           # )
            print(f"Merging LoRA from {self.model_name}...")
           # hacky change to fix loading
            model, tokenizer = FastVisionModel.from_pretrained(
                self.model_name,
                load_in_4bit=self.hyperparams['use_4bit'],
                load_in_8bit=self.hyperparams['use_8bit'],
                use_gradient_checkpointing="unsloth",
            )
            model.save_pretrained_merged(self.hyperparams.get('model_dir') + '/merged_base', tokenizer, save_method = "merged_16bit")
            print ("Merged model saved to "+ self.hyperparams.get('model_dir') + '/merged_base')
            #todo: check that this produces the same results
            model = model.merge_and_unload()
            print("Merge done.")
        else:
            model, tokenizer = FastVisionModel.from_pretrained(
                self.model_name,
                load_in_4bit=self.hyperparams['use_4bit'],
                load_in_8bit=self.hyperparams['use_8bit'],
                use_gradient_checkpointing="unsloth",
            )
        
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=True,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=self.hyperparams['lora_r'],
            lora_alpha=self.hyperparams['lora_r']/2,
            lora_dropout=self.hyperparams['lora_dropout'],
            use_rslora=self.hyperparams['use_rslora'],
            loftq_config=None,
        )

        training_args = SFTConfig(
            output_dir=self.hyperparams['model_dir'],
            per_device_train_batch_size=self.hyperparams['train_batch_size'],
            gradient_accumulation_steps=self.hyperparams['gradient_accumulation_steps'],
            warmup_ratio=self.hyperparams['warmup_ratio'],
            num_train_epochs=self.hyperparams['epochs'],
            learning_rate=self.hyperparams['learning_rate'],
            weight_decay=self.hyperparams['weight_decay'],
            seed=seed,
            optim="adamw_8bit",
            save_strategy="steps",
            save_steps=5000,
            logging_steps=100,
            eval_strategy="steps" if valid_samples else "no",
            eval_steps=5000, 
            load_best_model_at_end=True if valid_samples else False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb" if self.use_wandb else "none",
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=self.hyperparams['dataset_num_proc'],
            max_seq_length=self.hyperparams['max_seq_length'],
            dataset_text_field="",
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
        )

        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=False,
                max_pixels=self.hyperparams['max_pixels']
            )

        # This is too long to process!
        # cer_callback = CEREvalCallback(
        #     model=model,
        #     processor=self.processor,
        #     eval_samples=list(converted_valid_set)[:50] if converted_valid_set else [],
        #     device=self.device,
        # )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=converted_train_set,
            eval_dataset=converted_valid_set,
            data_collator=UnslothVisionDataCollator(model, self.processor),
            # callbacks=[cer_callback],
        )

        from transformers import EarlyStoppingCallback
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience = 200,   # How many steps we will wait if the eval loss doesn't decrease
            early_stopping_threshold = 0.01,  # Can set higher - sets how much loss should decrease by until
                                            # we consider early stopping.
        )

        trainer.add_callback(early_stopping_callback)

        print("Starting training...")
        checkpoint_dir = self.hyperparams['model_dir']
        has_checkpoint = any(
            Path(checkpoint_dir, d).is_dir()
            for d in os.listdir(checkpoint_dir)
            if d.startswith("checkpoint-")
        ) if os.path.exists(checkpoint_dir) else False

        trainer.train(resume_from_checkpoint=has_checkpoint if has_checkpoint else None)

        model_save_path = f"{training_args.output_dir}/{self.model_name.split('/')[-1]}-line-finetuned"
        print(f"Saving fine-tuned model to {model_save_path}")
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"Training complete! Model saved to {model_save_path}")

        config_path = self._create_finetuned_config(model_save_path, global_path, 'VLMLineHTR')
        
        print(f"\nTo run prediction with fine-tuned model:")
        print(f"   docworkflow -c {config_path} predict -t htr -d test")

        del model, tokenizer, trainer
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

    @abstractmethod
    def _process_batch(self, file_paths, source_dir, output_dir, save_image=True, **kwargs):
        """Must be implemented by subclasses (page vs line level)."""
        pass
