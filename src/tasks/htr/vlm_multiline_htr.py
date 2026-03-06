from src.tasks.htr.base_vlm_htr import BaseVLMHTR
import os
import shutil
import gc
import torch
import glob
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from lxml import etree as ET
import yaml

from src.alto.alto_lines import extract_lines_from_alto
from src.alto.alto_text import copy_alto_without_text

class _LazyLineDataset:
    def __init__(self, samples, format_fn):
        self.samples = samples
        self.format_fn = format_fn
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        result = self.format_fn(self.samples[idx])
        if result is None:
            raise ValueError(f"Invalid sample at index {idx}")
        return result

class VLMMultiLineHTRTask(BaseVLMHTR):
    """
    HTR using VLM for line-level transcription.
    Processes pre-segmented lines from ALTO XML files.
    Examples: Qwen3-VL-2B-catmus, Idefics3, etc.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "HTR_VLM_Line_Level"
        self.batch_size = config.get('line_batch_size', 1)

        self.hyperparams.update({
            'lora_r': config.get('lora_r', 16),
            'lora_dropout': config.get('lora_dropout', 0),
            'use_rslora': config.get('use_rslora', False),
            'max_seq_length': config.get('max_seq_length', 1024),
            'output_dir': config.get('output_dir', 'src/tasks/htr/models'),
            'train_batch_size': config.get('train_batch_size', 4),
            'gradient_accumulation_steps': config.get('gradient_accumulation_steps', 4),
            'warmup_ratio': config.get('warmup_ratio', 0.1),
            'epochs': config.get('epochs', 3),
            'learning_rate': config.get('learning_rate', 2e-4),
            'weight_decay': config.get('weight_decay', 0.01),
            'dataset_num_proc': config.get('dataset_num_proc', 1),
        })
    
    def _get_file_extensions(self):
        """VLM Line-Level works with ALTO XML files."""
        return ['*.xml']
    
    # modified for multiple images
    #TODO: modify args
    def _prepare_messages(self, imgs, prompt=None):
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
                    {"type": "text", "text": prompt}] + [{"type": "image", "image": img} for img in imgs]
                
            }
        ]
    #TODO: modify for multiple images
    # treat line_images as list of list of images
    def _recognize_batch(self, line_images):
        """
        Recognize multiple lines at once.
        
        Args:
            line_images: List of PIL Images (or None)
            
        Returns:
            List of recognition results
        """
        if self.is_minicpm:
            results = [{'text': '', 'confidence': 0.0} for _ in line_images]
            for i, imgs in enumerate(line_images):
                if imgs is None:
                    continue
                messages = self._prepare_messages(imgs) # need to define custom prepare_messages
                text = self._generate_from_messages(messages) # possibly fine as is?
                results[i] = {'text': text, 'confidence': 1.0}
            return results

        # Filter valid images
        #TODO: come back to some of this logic so we can do it on an image basis instead of set of 20 images
        # i might also need to associate indices with individual images
        # valid image checking should probably happen inside message preparation...? since we're batching
        valid_images = [(i, imgs) for i, imgs in enumerate(line_images) if imgs is not None]
        
        if not valid_images:
            results = [{'text': '', 'confidence': 0.0} for x in line_images for _ in x ]
        
        indices, images = zip(*valid_images)
        # going to ignore indices probably. will come back to do this more neatly
        
        # Prepare batch messages
        batch_messages = []
        for imgs in images:
            batch_messages.append(self._prepare_messages(imgs))
        #for m in batch_messages:
           # print(m)
        
        # Process batch using Qwen format
        inputs = self.processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        
        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        
        # Cleanup
        del inputs, generated_ids, generated_ids_trimmed
        
        # we're going to assume each line image output is separated by a newline
        texts_split=[x for y in output_texts for x in y.splitlines()]
        # Reconstruct results
        #TODO: I will need to be able to parse this as multiple lines....
        results = [{'text': '', 'confidence': 0.0} for _ in texts_split]
        #print(len(texts_split))
        #for idx, text in zip(indices, output_texts):
        for idx, text in enumerate(texts_split):
            results[idx] = {'text': text.strip(), 'confidence': 1.0}
        
        return results
        
    #TODO: modify for multiple images
    def _process_batch(self, file_paths, source_dir, output_dir, save_image=True, **kwargs):
        """
        Process ALTO files with line-level VLM.
        
        Args:
            file_paths: List of ALTO XML paths
            source_dir: Source directory
            output_dir: Output directory
            save_image: Whether to copy images
            
        Returns:
            List of results
        """
        print(f"  Processing {len(file_paths)} ALTO files...")
        
        results = []

        for alto_path in tqdm(file_paths, desc="  Recognizing lines", unit="page"):
            try:
                image_path, lines, _ = extract_lines_from_alto(alto_path)
                
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
                
                # Extract all line images
                # Modify to do batches of batches
                # we want groups of up to max 20 lines maybe?
                maxlines = 40 #TODO: make this a hyperparameter we can tune. setting to 1 should function as regular vlm line
                nlines = len(lines)

                line_images = []
                lineblocks = [lines[i:i+maxlines] for i in range(0,nlines, maxlines)]
                for block in lineblocks:
                    block_images = []
                    for line in block:
                        if line.get('boundary'):
                            line_img = self._extract_line_image(page_image, line['boundary'])
                            block_images.append(line_img)
                        else:
                            block_images.append(None)
                    line_images.append(block_images)
                
                # Process in batches
                recognized_texts = []
                for i in range(0, len(line_images), self.batch_size):
                    batch = line_images[i:i+self.batch_size]
                    batch_results = self._recognize_batch(batch)
                    recognized_texts.extend(batch_results)
                
                # Save to ALTO
                output_path = os.path.join(output_dir, os.path.basename(alto_path))
                
                if not os.path.exists(output_path):
                    copy_alto_without_text(alto_path, output_path)
                
                self._add_text_to_alto(output_path, recognized_texts, output_path)
                
                results.append({'file': alto_path, 'texts': recognized_texts})
                
                # Copy image
                if save_image:
                    image_output = os.path.join(output_dir, os.path.basename(image_path))
                    if not os.path.exists(image_output):
                        shutil.copy2(image_path, image_output)
                
                page_image.close()
                
                # Memory cleanup
                if len(results) % 5 == 0:
                    gc.collect()
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  Error processing {alto_path}: {e}")
                import traceback
                traceback.print_exc()
        
        return results

    def _add_text_to_alto(self, alto_path, texts, output_path):
        """Add recognized text to ALTO XML."""
        tree = ET.parse(alto_path)
        root = tree.getroot()
        ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
        
        text_lines = root.findall('.//alto:TextLine', ns)
        
        for line, text_data in zip(text_lines, texts):
            if text_data and 'text' in text_data and text_data['text']:
                # Remove existing String elements
                for string_elem in line.findall('alto:String', ns):
                    line.remove(string_elem)
                
                # Add new String
                string_elem = ET.SubElement(line, f"{{{ns['alto']}}}String")
                string_elem.set('CONTENT', text_data['text'])
                string_elem.set('WC', str(text_data.get('confidence', 1.0)))
        
        tree.write(output_path, pretty_print=True, 
                  xml_declaration=True, encoding="UTF-8")
        

    def _prepare_training_data_lines(self, data_path):
        """
        Prepare line-level training samples from ALTO XML + page images.
        Extracts each TextLine bbox and its ground truth text.
        Crops are performed lazily in format_conversation.
        
        Returns:
            List of dicts with page_image_path, text, bbox (left, top, right, bottom)
        """

        samples = []
        xml_files = glob.glob(os.path.join(str(data_path), "**", "*.xml"), recursive=True)
        skipped = 0

        for xml_path in xml_files:
            base_name = Path(xml_path).stem
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                p = os.path.join(os.path.dirname(xml_path), base_name + ext)
                if os.path.exists(p):
                    image_path = p
                    break

            if not image_path:
                skipped += 1
                continue

            # for line in extract_lines_with_bbox_from_alto(xml_path):
            #     left  = line['hpos']
            #     top   = line['vpos']
            #     right = line['hpos'] + line['width']
            #     bottom = line['vpos'] + line['height']
            #     if right <= left or bottom <= top:
            #         continue
            #     samples.append({
            #         "page_image_path": image_path,
            #         "text": line['text'],
            #         "bbox": (left, top, right, bottom),
            #     })
            _, lines, _ = extract_lines_from_alto(xml_path)
            # we want groups of up to max 20 lines maybe?
            maxlines = 40 #TODO: make this a hyperparameter we can tune. setting to 1 should function as regular vlm line
            nlines = len(lines)
            lineblocks = [lines[i:i+maxlines] for i in range(0,nlines, maxlines)]
            for block in lineblocks:
                samples.append({ 
                    "page_image_path": image_path,
                    "text": [line['text'] for line in block],
                    "boundary": [line['boundary'] for line in block]
                })

        if skipped:
            print(f"  Warning: {skipped} XML files had no matching image")
        print(f"  Extracted {len(samples)} line samples from {len(xml_files) - skipped} pages")
        return samples

    def train(self, data_path=None, seed=42):
        import unsloth
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

        global_path = str(data_path.parent)

        print(f"Starting VLM line-level fine-tuning with Unsloth")
        print(f"Model: {self.model_name}")

        print("Preparing line-level training data...")
        train_samples = self._prepare_training_data_lines(data_path)
        valid_samples = self._prepare_training_data_lines(global_path + "/valid")

        if not train_samples:
            raise ValueError("No valid line-level training samples found")

        print(f"Found {len(train_samples)} line samples (train) and {len(valid_samples)} (valid)")

        def format_conversation(example): 
            try:
                page_img = Image.open(example["page_image_path"]).convert("RGB")
                imgs = [self._extract_line_image(page_img, b) for b in example["boundary"]]
            except Exception as e:
                print(f"Warning: skipping sample ({example.get('page_image_path', '?')}): {e}")
                return None
            #TODO: better error handling for if only one crop is bad
            if imgs is None:
                return None

            # modified to make conversation give a list of images and return a list of texts
            # it's possible this will be confusing for the model but it's also possible more context is helpful
            # check to see if what we get in return has newlines or if i need to put them in for formatting purposes
            return {"messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt}]+[{"type": "image", "image": img} for img in imgs]
                    
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": t} for t in example["text"]],
                },
            ]}

        print("Validating train samples...")
        valid_train = [s for s in train_samples if os.path.exists(s["page_image_path"]) and s.get("boundary")]
        if len(valid_train) < len(train_samples):
            print(f"  Skipped {len(train_samples) - len(valid_train)} samples")
        converted_train_set = _LazyLineDataset(valid_train, format_conversation)

        if valid_samples:
            valid_valid = [s for s in valid_samples if os.path.exists(s["page_image_path"]) and s.get("boundary")]
            converted_valid_set = _LazyLineDataset(valid_valid, format_conversation)
        else:
            converted_valid_set = None

        print("Loading model with Unsloth...")
        model, tokenizer = FastVisionModel.from_pretrained(
            self.model_name,
            load_in_4bit=self.hyperparams['use_4bit'],
            use_gradient_checkpointing="unsloth",
        )

        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=True,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=self.hyperparams['lora_r'],
            lora_alpha=self.hyperparams['lora_r'],
            lora_dropout=self.hyperparams['lora_dropout'],
            use_rslora=self.hyperparams['use_rslora'],
            loftq_config=None,
        )

        training_args = SFTConfig(
            output_dir=self.hyperparams['output_dir'],
            per_device_train_batch_size=self.hyperparams['train_batch_size'],
            gradient_accumulation_steps=self.hyperparams['gradient_accumulation_steps'],
            warmup_ratio=self.hyperparams['warmup_ratio'],
            num_train_epochs=self.hyperparams['epochs'],
            learning_rate=self.hyperparams['learning_rate'],
            weight_decay=self.hyperparams['weight_decay'],
            seed=seed,
            optim="adamw_8bit",
            save_strategy="steps",
            save_steps=500,
            logging_steps=10,
            eval_strategy="steps" if valid_samples else "no",
            eval_steps=100, 
            load_best_model_at_end=True if valid_samples else False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb" if self.use_wandb else "none",
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=self.hyperparams['dataset_num_proc'],
            max_seq_length=self.hyperparams['max_seq_length'],
            dataset_text_field="",
        )

        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=False,
                max_pixels=self.hyperparams['max_pixels']
            )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=converted_train_set,
            eval_dataset=converted_valid_set,
            data_collator=UnslothVisionDataCollator(model, self.processor),
        )

        print("Starting training...")
        trainer.train()

        model_save_path = f"{training_args.output_dir}/{self.model_name.split('/')[-1]}-line-finetuned"
        print(f"Saving fine-tuned model to {model_save_path}")
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"Training complete! Model saved to {model_save_path}")

        config_path = self._create_finetuned_config(model_save_path, global_path)
        
        print(f"\nTo run prediction with fine-tuned model:")
        print(f"   docworkflow -c {config_path} predict -t htr -d test")

        del model, tokenizer, trainer
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

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
                    'type': 'VLMMultiLineHTR',
                    'config': {
                        'model_name': output_dir,
                        'base_model': self.model_name,
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