from src.tasks.htr.base_htr import BaseHTR

import os
import glob
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import shutil
import torch
import gc
import yaml

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

        self.model_name = config.get('model_name', 'stanford-oval/churro-3B')
        self.name = "HTR_" + self.model_name.split('/')[-1]
        self.max_new_tokens = config.get('max_new_tokens', 512)
        self.batch_size = config.get('batch_size', 1)
        
        self.prompt = config.get('prompt',
                                 "You are an expert in transcription of historical documents from various languages. "
                                 "Your task is to extract the full text from a given page."
                                 "Use exactly ONE line break between each line of text."
                                )

        # Store all hyperparameters in a dictionary
        self.hyperparams = {
            # Model loading configuration
            'use_dtype_param': config.get('use_dtype_param', False),
            'device_map': config.get('device_map'),
            'attn_implementation': config.get('attn_implementation'),
            'model_class': config.get('model_class', None),

            # Training hyperparameters
            'use_4bit': config.get('use_4bit', False),
            'lora_r': config.get('lora_r', 16),
            'lora_dropout': config.get('lora_dropout', 0),
            'use_rslora': config.get('use_rslora', False),
            'max_seq_length': config.get('max_seq_length', 4096),
            'output_dir': config.get('output_dir', f"src/tasks/htr/models"),
            'train_batch_size': config.get('train_batch_size', 1),
            'gradient_accumulation_steps': config.get('gradient_accumulation_steps', 4),
            'warmup_ratio': config.get('warmup_ratio', 0.1),
            'epochs': config.get('epochs', 3),
            'learning_rate': config.get('learning_rate', 2e-4),
            'weight_decay': config.get('weight_decay', 0.01),
            'dataset_num_proc': config.get('dataset_num_proc', 1)
        }

        self.processor = None
        self.model = None

    def load(self):
        """
        Load the VLM model from Transformers.
        Supports both full models and LoRA adapters.
        """
        
        print(f"Loading VLM model: {self.model_name}")
        print("This may take several minutes on first run...")

        use_lora = self.config.get('use_lora_adapter', False)
        
        if use_lora: 
            from peft import PeftModel
            
            base_model_name = self.config.get('base_model_name')
            if not base_model_name:
                raise ValueError("base_model_name must be specified when use_lora_adapter=true")
            
            print(f"Loading base model: {base_model_name}")
            print(f"Loading LoRA adapter from: {self.model_name}")
            
            # Load processor from adapter directory or base model
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
            except:
                self.processor = AutoProcessor.from_pretrained(
                    base_model_name,
                    trust_remote_code=True
                )

            # Configuration for base model loading
            model_kwargs = {
                "trust_remote_code": True,
            }
            
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

            # Load base model
            model_class_name = self.hyperparams['model_class']
            
            if model_class_name == 'AutoModel':
                base_model = AutoModel.from_pretrained(base_model_name, **model_kwargs)
            elif model_class_name == 'AutoModelForImageTextToText':
                base_model = AutoModelForImageTextToText.from_pretrained(base_model_name, **model_kwargs)
            else:
                if is_supported_by_auto_image_text(base_model_name):
                    base_model = AutoModelForImageTextToText.from_pretrained(base_model_name, **model_kwargs)
                else:
                    base_model = AutoModel.from_pretrained(base_model_name, **model_kwargs)
            
            # Load LoRA adapter
            print("Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(
                base_model,
                self.model_name,
                is_trainable=False
            )
            
        else:
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
                if self.hyperparams['use_dtype_param']:
                    model_kwargs['dtype'] = torch.bfloat16
                else:
                    model_kwargs['torch_dtype'] = torch.bfloat16
            else:
                if self.hyperparams['use_dtype_param']:
                    model_kwargs['dtype'] = torch.float32
                else:
                    model_kwargs['torch_dtype'] = torch.float32

            # Add optional config parameters
            if self.hyperparams['device_map']:
                model_kwargs['device_map'] = self.hyperparams['device_map']

            if self.hyperparams['attn_implementation']:
                model_kwargs['attn_implementation'] = self.hyperparams['attn_implementation']

            # Determine which Auto class to use
            model_class_name = self.hyperparams['model_class']
            
            if model_class_name == 'AutoModel':
                print("Using AutoModel (explicitly specified)")
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            elif model_class_name == 'AutoModelForImageTextToText':
                print("Using AutoModelForImageTextToText (explicitly specified)")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            else:
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

    def _split_output_into_lines(self, alto_path, text, image_path):
        """
        Split single-line output into multiple TextLines based on line breaks.
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
                self._split_output_into_lines(output_path, text, image_path)
                
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
    

    def train(self, data_path=None, seed=42):
        """
        Fine-tune the VLM model using Unsloth.
        
        Args:
            data_path: Path to training data (directory with images and ALTO XML files)
            seed: Random seed for reproducibility
        """       

        print(f"To train this model, you must change the environnement to vlm-training:")
        print(f"\n  source envs/vlm-training/bin/activate")        
        response = input("Would you like to launch training now? (y/n): ")

        if response.lower() == 'y':
            from unsloth import FastVisionModel
            from unsloth.trainer import UnslothVisionDataCollator
            from transformers import AutoProcessor
            from trl import SFTTrainer, SFTConfig
            from datasets import Dataset

            if not data_path:
                raise ValueError("Training data path is required")
            
            print(f"Starting VLM fine-tuning with Unsloth")
            print(f"Model: {self.model_name}")
            
            print("Preparing training data...")
            samples = self._prepare_training_data(data_path)
            
            if not samples:
                raise ValueError("No valid training samples found")
            
            print(f"Found {len(samples)} training samples")


            def format_conversation(example):
                img = Image.open(example["image_path"]).convert("RGB")

                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prompt},
                            {"type": "image", "image": img},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": example["text"]}
                        ],
                    },
                ]
                
                return {"messages": conversation}

            converted_dataset = [format_conversation(sample) for sample in samples]


            print("Loading model with Unsloth...")
            model, tokenizer = FastVisionModel.from_pretrained(
                self.model_name,
                load_in_4bit=self.hyperparams['use_4bit'],
                use_gradient_checkpointing="unsloth",
            )

            model = FastVisionModel.get_peft_model(
                model,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
                r=self.hyperparams['lora_r'],
                lora_alpha=self.hyperparams['lora_r'],
                lora_dropout=self.hyperparams['lora_dropout'],
                use_rslora=self.hyperparams['use_rslora'],
                loftq_config=None,
            )

            # Training arguments
            training_args = SFTConfig(
                output_dir=self.hyperparams['output_dir'],
                per_device_train_batch_size=self.hyperparams['train_batch_size'],
                gradient_accumulation_steps=self.hyperparams['gradient_accumulation_steps'],
                warmup_ratio=self.hyperparams['warmup_ratio'],
                num_train_epochs=self.hyperparams['epochs'],
                learning_rate=self.hyperparams['learning_rate'],
                weight_decay=self.hyperparams['weight_decay'],
                seed=seed,
                save_strategy="steps",
                save_steps=500,
                logging_steps=10,
                report_to="wandb" if self.use_wandb else "none",
                remove_unused_columns=False,
                dataset_kwargs = {"skip_prepare_dataset": True},
                dataset_num_proc = self.hyperparams['dataset_num_proc'],
                max_seq_length = self.hyperparams['max_seq_length'],
                dataset_text_field="",
            )
            
            if self.processor is None:
                print("Loading processor for data preparation...")
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    use_fast=False
                )
                        
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=converted_dataset,
                data_collator=UnslothVisionDataCollator(model, self.processor),
            )
            
            print("Starting training...")
            trainer.train()
            
            output_dir = training_args.output_dir
            model_save_path = f"{output_dir}/{self.model_name.split('/')[-1]}-finetuned"
            
            print(f"Saving fine-tuned model to {model_save_path}")
            
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            
            print(f"Training complete! Model saved to {model_save_path}")
            
            # Create inference config
            print("\nCreating inference configuration...")
            config_path = self._create_finetuned_config(model_save_path)
            
            print(f"\nTo run prediction with fine-tuned model:")
            print(f"   docworkflow -c {config_path} predict -t htr -d test")
            
            # Cleanup
            del model, tokenizer, trainer
            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()


    def _prepare_training_data(self, data_path):
        """
        Prepare training data with conversation structure.
        Returns a list of image paths and texts, images will be loaded on-the-fly.
        """
        samples = []
        xml_files = glob.glob(os.path.join(data_path, "*.xml"))
        
        for xml_path in xml_files:
            # Extract text
            text = self._extract_text_from_alto(xml_path)
            if not text or not text.strip():
                continue
            
            # Find image
            base_name = Path(xml_path).stem
            image_path = None
            
            for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                potential_path = os.path.join(data_path, base_name + ext)
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
            
            if not image_path:
                print(f"Warning: No image found for {xml_path}")
                continue
            
            # Store the path for now
            samples.append({
                "image_path": image_path,
                "text": text
            })
        
        return samples
    
    def _create_finetuned_config(self, output_dir, original_config_path=None):
        """
        Create a configuration file for the fine-tuned model.
        
        Args:
            output_dir: Directory where the model was saved
            original_config_path: Path to the original training config (optional)
        
        Returns:
            Path to the created config file
        """
        
        model_name = self.model_name.split('/')[-1]
        finetuned_name = f"{model_name}-finetuned"
        
        # Create config for the fine-tuned model
        config = {
            'run_name': f"{self.config.get('run_name', 'model')}_finetuned",
            'output_dir': 'results',
            'device': self.config.get('device', 'cuda'),
            'use_wandb': self.config.get('use_wandb', False),
            'wandb_project': self.config.get('wandb_project', 'HTR-comparison'),
            'data': {
                'test': self.config.get('data', {}).get('test', '../data/test/')
            },
            'tasks': {
                'htr': {
                    'type': 'VLMHTR',
                    'config': {
                        'model_name': output_dir,
                        'base_model_name': self.model_name,
                        'use_lora_adapter': True,
                        'max_new_tokens': self.max_new_tokens,
                        'batch_size': self.batch_size,
                        'prompt': self.prompt,
                        **{k: v for k, v in self.hyperparams.items() 
                        if k in ['use_dtype_param', 'device_map', 'attn_implementation', 'model_class']
                        and v is not None}
                    }
                }
            }
        }
        
        # Save config in the model directory
        model_config_path = Path(output_dir) / 'inference_config.yml'
        with open(model_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n Inference config saved to: {model_config_path}")
        
        return model_config_path