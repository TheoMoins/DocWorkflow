from src.tasks.htr.base_vlm_htr import BaseVLMHTR
import os
import glob
import shutil
import gc
import torch
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from lxml import etree as ET
import yaml

from src.alto.alto_text import copy_and_fix_alto_namespaces


class VLMPageHTRTask(BaseVLMHTR):
    """
    HTR using VLM for page-level transcription.
    Processes entire page images and splits output into lines.
    Examples: CHURRO, etc.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "HTR_VLM_Page_Level"
    
    def _process_batch(self, file_paths, source_dir, output_dir, save_image=True, **kwargs):
        """
        Process images for page-level VLM HTR.
        
        Args:
            file_paths: List of image paths
            source_dir: Source directory
            output_dir: Output directory
            save_image: Whether to copy images
            
        Returns:
            List of results
        """
        print(f"  Processing {len(file_paths)} images...")
        
        results = []
        
        for image_path in tqdm(file_paths, desc="  Recognizing text", unit="image"):
            try:
                # Recognize full page
                messages = self._prepare_messages(Image.open(image_path).convert("RGB"))
                text = self._generate_from_messages(messages)
                
                # Create/update ALTO
                base_name = Path(image_path).stem
                output_path = os.path.join(output_dir, f"{base_name}.xml")
                
                existing_alto = os.path.join(source_dir, f"{base_name}.xml")
                if os.path.exists(existing_alto):
                    copy_and_fix_alto_namespaces(existing_alto, output_path)
                else:
                    self._create_simple_alto_with_text(image_path, text, output_path)
                
                # Split into lines
                self._split_output_into_lines(output_path, text, image_path)
                
                results.append({'file': image_path, 'text': text})
                
                if save_image:
                    image_output = os.path.join(output_dir, os.path.basename(image_path))
                    if not os.path.exists(image_output):
                        shutil.copy2(image_path, image_output)
                
                # Memory cleanup
                if len(results) % 10 == 0:
                    gc.collect()
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  Error processing {image_path}: {e}")
                import traceback
                traceback.print_exc()
        
        return results
    
    def _split_output_into_lines(self, alto_path, text, image_path):
        """Split VLM output into TextLines (same as before)."""
        from PIL import Image
        
        lines_text = [line.strip() for line in text.split('\n') if line.strip()]
        
        if not lines_text:
            return alto_path
        
        tree = ET.parse(alto_path)
        root = tree.getroot()
        ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
        
        with Image.open(image_path) as img:
            width, height = img.size
        
        existing_lines = root.findall('.//alto:TextLine', ns)
        
        if existing_lines and len(existing_lines) == len(lines_text):
            # Perfect match
            for line_elem, line_text in zip(existing_lines, lines_text):
                for string_elem in line_elem.findall('alto:String', ns):
                    line_elem.remove(string_elem)
                
                string_elem = ET.SubElement(line_elem, ET.QName(ns['alto'], 'String'))
                string_elem.set('CONTENT', line_text)
                string_elem.set('WC', '1.0')
        else:
            # Create new structure
            text_blocks = root.findall('.//alto:TextBlock', ns)
            
            if text_blocks:
                text_block = text_blocks[0]
                for extra_block in text_blocks[1:]:
                    extra_block.getparent().remove(extra_block)
                
                for line_elem in text_block.findall('alto:TextLine', ns):
                    text_block.remove(line_elem)
                
                line_height = height // max(len(lines_text), 1)
                margin = 10
                
                for idx, line_text in enumerate(lines_text):
                    y_pos = idx * line_height + margin
                    line_elem = ET.SubElement(text_block, ET.QName(ns['alto'], 'TextLine'))
                    line_elem.set('ID', f'line_{idx}')
                    line_elem.set('HPOS', str(margin))
                    line_elem.set('VPOS', str(y_pos))
                    line_elem.set('WIDTH', str(width - 2 * margin))
                    line_elem.set('HEIGHT', str(line_height - margin))
                    
                    baseline_y = y_pos + line_height // 2
                    line_elem.set('BASELINE', f"{margin} {baseline_y} {width - margin} {baseline_y}")
                    
                    string_elem = ET.SubElement(line_elem, ET.QName(ns['alto'], 'String'))
                    string_elem.set('CONTENT', line_text)
                    string_elem.set('WC', '1.0')
        
        def indent_xml(elem, level=0):
            i = "\n" + level*"  "
            if len(elem):
                if not elem.text or not elem.text.strip():
                    elem.text = i + "  "
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
                for child in elem:
                    indent_xml(child, level+1)
                if not child.tail or not child.tail.strip():
                    child.tail = i
            else:
                if level and (not elem.tail or not elem.tail.strip()):
                    elem.tail = i
        
        indent_xml(root)
        ET.cleanup_namespaces(root)
        tree.write(alto_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")
        return alto_path



    def _process_batch(self, file_paths, source_dir, output_dir, save_image=True, **kwargs):
        """
        Process a batch of images for VLM HTR.
        
        Args:
            file_paths: List of image paths to process
            source_dir: Source directory (for finding ALTO if exists)
            output_dir: Directory to save ALTO XML files
            save_image: Whether to copy images to output
            
        Returns:
            List of prediction results
        """
        print(f"  Processing {len(file_paths)} images...")
        
        results = []
        
        for image_path in tqdm(file_paths, desc="  Recognizing text", unit="image"):
            try:
                # Recognize text
                text = self._recognize_single_image(image_path)
                
                # Create basic ALTO file
                base_name = Path(image_path).stem
                output_path = os.path.join(output_dir, f"{base_name}.xml")
                
                # Check if there's an existing ALTO with layout/lines
                existing_alto = os.path.join(source_dir, f"{base_name}.xml")
                if os.path.exists(existing_alto):
                    # Copy existing structure AND clean namespaces
                    copy_and_fix_alto_namespaces(existing_alto, output_path)
                else:
                    # Create simple ALTO
                    self._create_simple_alto_with_text(image_path, text, output_path)
                
                # Split VLM output into lines
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
                print(f"  Error processing {image_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return results
    

    def _recognize_single_image(self, image):
        messages = self._prepare_messages(image)
        return self._generate_from_messages(messages)

    def _process_batch(self, file_paths, source_dir, output_dir, save_image=True, **kwargs):
        """
        Process a batch of images for VLM HTR.
        
        Args:
            file_paths: List of image paths to process
            source_dir: Source directory (for finding ALTO if exists)
            output_dir: Directory to save ALTO XML files
            save_image: Whether to copy images to output
            
        Returns:
            List of prediction results
        """
        print(f"  Processing {len(file_paths)} images...")
        
        results = []
        
        for image_path in tqdm(file_paths, desc="  Recognizing text", unit="image"):
            try:
                # Recognize text
                text = self._recognize_single_image(image_path)
                
                # Create basic ALTO file
                base_name = Path(image_path).stem
                output_path = os.path.join(output_dir, f"{base_name}.xml")
                
                # Check if there's an existing ALTO with layout/lines
                existing_alto = os.path.join(source_dir, f"{base_name}.xml")
                if os.path.exists(existing_alto):
                    # Copy existing structure AND clean namespaces
                    copy_and_fix_alto_namespaces(existing_alto, output_path)
                else:
                    # Create simple ALTO
                    self._create_simple_alto_with_text(image_path, text, output_path)
                
                # Split VLM output into lines
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
                print(f"  Error processing {image_path}: {e}")
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

        from unsloth import FastVisionModel
        from unsloth.trainer import UnslothVisionDataCollator
        from transformers import AutoProcessor
        from trl import SFTTrainer, SFTConfig

        if not data_path:
            raise ValueError("Training data path is required")
        
        global_path = str(data_path.parent)     

        print(f"Starting VLM fine-tuning with Unsloth")
        print(f"Model: {self.model_name}")
        
        print("Preparing training data...")
        train_samples = self._prepare_training_data(data_path)
        valid_samples = self._prepare_training_data(global_path+"/valid")
        
        if not train_samples:
            raise ValueError("No valid training samples found")
        
        print(f"Found {len(train_samples)} training samples and {len(valid_samples)} validation samples")


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

        converted_train_set = [format_conversation(sample) for sample in train_samples]
        converted_valid_set = [format_conversation(sample) for sample in valid_samples]


        print("Loading model with Unsloth...")
        model, tokenizer = FastVisionModel.from_pretrained(
            self.model_name,
            load_in_4bit=self.hyperparams['use_4bit'],
            use_gradient_checkpointing="unsloth",
        )

        model = FastVisionModel.get_peft_model(
            model,
            
            finetune_vision_layers     = True, # False if not finetuning vision layers
            finetune_language_layers   = True, # False if not finetuning language layers
            finetune_attention_modules = True, # False if not finetuning attention layers
            finetune_mlp_modules       = True, # False if not finetuning MLP layers 
                            
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
            optim = "adamw_8bit",
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
                    
        # TEST: Vérifier les données converties
        print("\nTesting converted datasets...")
        print(f"Train dataset size: {len(converted_train_set)}")
        print(f"Valid dataset size: {len(converted_valid_set)}")
        print(f"First train sample keys: {converted_train_set[0].keys()}")

        # TEST: Vérifier qu'on peut charger la première image
        try:
            first_sample = converted_train_set[0]
            messages = first_sample['messages']
            user_content = messages[0]['content']
            for item in user_content:
                if item['type'] == 'image':
                    img = item['image']
                    print(f"✓ First image loaded: {img.size if hasattr(img, 'size') else type(img)}")
                    break
        except Exception as e:
            print(f"✗ Error with first sample: {e}")
            import traceback
            traceback.print_exc()

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=converted_train_set,
            eval_dataset=converted_valid_set,
            data_collator=UnslothVisionDataCollator(model, self.processor),
        )
        
        trainer.train()
        
        output_dir = training_args.output_dir
        model_save_path = f"{output_dir}/{self.name.split('/')[-1]}-finetuned"
        
        print(f"Saving fine-tuned model to {model_save_path}")
        
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        
        print(f"Training complete! Model saved to {model_save_path}")
        
        config_path = self._create_finetuned_config(model_save_path, global_path)
        
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
    
    def _create_finetuned_config(self, output_dir, global_path):
        """
        Create a configuration file for the fine-tuned model.
        
        Args:
            output_dir: Directory where the model was saved
            original_config_path: Path to the original training config (optional)
        
        Returns:
            Path to the created config file
        """

        # Create config for the fine-tuned model
        config = {
            'run_name': f"{self.name}_finetuned",
            'output_dir': 'results',
            'device': self.config.get('device', 'cuda'),
            'use_wandb': self.config.get('use_wandb', False),
            'wandb_project': self.config.get('wandb_project', 'HTR-comparison'),
            'data': {
                'train': global_path + '/train',
                'valid': global_path + '/valid',
                'test': global_path + '/test'
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