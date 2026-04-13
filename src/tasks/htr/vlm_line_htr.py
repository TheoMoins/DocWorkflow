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
from peft import PeftModel


from transformers import TrainerCallback

from src.tasks.htr.prompt_convention import build_conventions_block, load_conventions
from src.content.weighted_sampling import special_char_density
from src.alto.alto_lines import extract_lines_from_alto
from src.alto.alto_text import copy_alto_without_text
from src.utils.lazy_dataset import LazyLineDataset as _LazyLineDataset

Image.MAX_IMAGE_PIXELS = None
    
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
                    enable_thinking=False,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,          # greedy pour le CER
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


class VLMLineHTRTask(BaseVLMHTR):
    """
    HTR using VLM for line-level transcription.
    Processes pre-segmented lines from ALTO XML files.
    Examples: Qwen3-VL-2B-catmus, Idefics3, etc.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "HTR_VLM_Line_Level"
        self.batch_size = config.get('line_batch_size', 1)
        self.train_sources = config.get('train_sources', None)
        self.prompt_template = config.get('prompt_template', self.prompt)

        self.hyperparams.update({
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
        })
    
    def _get_file_extensions(self):
        """VLM Line-Level works with ALTO XML files."""
        return ['*.xml']
    
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
            for i, img in enumerate(line_images):
                if img is None:
                    continue
                messages = self._prepare_messages(img)
                text = self._generate_from_messages(messages)
                results[i] = {'text': text, 'confidence': 1.0}
            return results

        # Filter valid images
        valid_images = [(i, img) for i, img in enumerate(line_images) if img is not None]
        
        if not valid_images:
            return [{'text': '', 'confidence': 0.0} for _ in line_images]
        
        indices, images = zip(*valid_images)
        
        # Prepare batch messages
        batch_messages = []
        for img in images:
            batch_messages.append(self._prepare_messages(img))
        
        # Process batch using Qwen format
        inputs = self.processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            padding=True,
            enable_thinking=False,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            gen_kwargs = self._build_base_gen_kwargs()
            gen_kwargs.update({"max_new_tokens": self.max_new_tokens, "do_sample": False})
            generated_ids = self.model.generate(**inputs, **gen_kwargs)        

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
        
        # Reconstruct results
        results = [{'text': '', 'confidence': 0.0} for _ in line_images]
        for idx, text in zip(indices, output_texts):
            results[idx] = {'text': text.strip(), 'confidence': 1.0}
        
        return results

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
                doc_samples = self._prepare_training_data_lines(doc_path)
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
            samples = self._prepare_training_data_lines(data_path)
            for s in samples:
                s['prompt'] = resolved_prompt
            conv_info = f"conventions: {list(conventions.keys())}" if conventions else "no conventions"
            print(f"  {data_path}: {len(samples)} samples ({conv_info})")

        return samples

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
                doc_dir = Path(alto_path).parent
                conventions = load_conventions(doc_dir)
                prompt_tpl = self.prompt_template
                if conventions and '{conventions}' in prompt_tpl:
                    self.prompt = prompt_tpl.replace('{conventions}', build_conventions_block(conventions))
                else:
                    self.prompt = prompt_tpl.replace('{conventions}', '').strip()

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
                line_images = []
                for line in lines:
                    if line.get('boundary'):
                        line_img = self._extract_line_image(page_image, line['boundary'])
                        line_images.append(line_img)
                    else:
                        line_images.append(None)
                
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
            for line in lines:
                samples.append({
                    "page_image_path": image_path,
                    "text": line['text'],
                    "boundary": line['boundary']
                })

        if skipped:
            print(f"  Warning: {skipped} XML files had no matching image")
        print(f"  Extracted {len(samples)} line samples from {len(xml_files) - skipped} pages")
        return samples

    def _format_conversation(self, example, get_page_image):
        try:
            page_img = get_page_image(example["page_image_path"])
            img = self._extract_line_image(page_img, example["boundary"])
        except Exception as e:
            print(f"Warning: skipping sample ({example.get('page_image_path', '?')}): {e}")
            return None
        if img is None:
            return None
        prompt = example.get("prompt", self.prompt)
        return {"messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": img},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example["text"]}],
            },
        ]}

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
        valid_train = [s for s in train_samples if os.path.exists(s["page_image_path"]) and s.get("boundary")]
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
            print(f"\n📊 Density of special caracters:")
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

        converted_train_set = _LazyLineDataset(valid_train, self._format_conversation)

        if valid_samples:
            valid_valid = [s for s in valid_samples if os.path.exists(s["page_image_path"]) and s.get("boundary")]
            converted_valid_set = _LazyLineDataset(valid_valid, self._format_conversation)
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
