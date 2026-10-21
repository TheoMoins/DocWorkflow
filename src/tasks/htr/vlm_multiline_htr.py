from src.tasks.htr.base_vlm_htr import BaseVLMHTR, bounding_box, pad_to_max_ratio
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

from src.tasks.htr.prompt_convention import build_conventions_block, load_conventions
from src.alto.alto_lines import read_lines_geometry
from src.alto.alto_text import copy_alto_without_text, write_text_to_alto
from src.utils.lazy_dataset import LazyLineDataset as _LazyLineDataset
from src.content.weighted_sampling import special_char_density


def group_lines_into_blocks(lines, k):
    """
    Groupe les lignes consécutives (ordre de lecture) par blocs d'au plus k lignes,
    sans jamais faire chevaucher deux TextBlocks : un changement de zone clôt le bloc
    courant, même s'il contient moins de k lignes.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    blocks = []
    current = []
    for line in lines:
        if current and (len(current) == k or line.get('block_id') != current[-1].get('block_id')):
            blocks.append(current)
            current = []
        current.append(line)
    if current:
        blocks.append(current)
    return blocks

def _normalized_edit_distance(a, b):
    """Distance de Levenshtein normalisée par la longueur de la plus longue chaîne."""
    if a == b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def parse_multiline_output(raw_text, k, dedup_threshold=0.1):
    """
    Découpe la sortie brute du modèle pour un bloc de k lignes.

    Étapes : split sur '\n', dédoublonnage des lignes consécutives quasi identiques
    (distance d'édition normalisée < dedup_threshold, on garde la première), puis
    alignement sur k lignes (complété par des chaînes vides ou tronqué).
    Les lignes vides ne sont jamais considérées comme des doublons.
    # Pour un jeton séparateur dédié, remplacer le split('\n') ci-dessous.

    Returns:
        (texts, n_mismatch, n_deduped) — texts est toujours de longueur k ;
        n_mismatch vaut 1 si le compte de lignes différait de k après dédoublonnage, sinon 0.
    """
    texts = []
    n_deduped = 0
    for line in raw_text.split('\n'):
        line = line.strip()
        if texts and line and texts[-1] and _normalized_edit_distance(texts[-1], line) < dedup_threshold:
            n_deduped += 1
            print(f"  [dedup] collapsed {line!r} into previous line {texts[-1]!r}")
            continue
        texts.append(line)

    n_mismatch = 0
    if len(texts) != k:
        n_mismatch = 1
        texts = texts[:k] + [''] * (k - len(texts))
    return texts, n_mismatch, n_deduped


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
        self.prompt_template = config.get('prompt_template', self.prompt)
        self.lines_per_crop = config.get('lines_per_crop', 4)
        self.rolling_window = config.get('rolling_window', False)
        self.dedup_threshold = config.get('dedup_threshold', 0.1)

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
            # An epoch is only ~150k blocks / batch (vs ~600k lines at k=1): keep the
            # steps configurable so short runs still evaluate and checkpoint.
            'save_steps': config.get('save_steps', 5000),
            'eval_steps': config.get('eval_steps', 5000),
            'logging_steps': config.get('logging_steps', 100),
        })
    
    def _get_file_extensions(self):
        """VLM Line-Level works with ALTO XML files."""
        return ['*.xml']

    def _extract_multiline_image(self, image, boundaries):
        """
        Crop le rectangle englobant l'union des polygones de plusieurs lignes.
        Contrairement à _extract_line_image, aucun masque polygonal n'est appliqué.
        """
        points = [pt for boundary in boundaries if boundary for pt in boundary]
        if not points:
            return None

        min_x, min_y, max_x, max_y = bounding_box(points)
        if max_x - min_x <= 0 or max_y - min_y <= 0:
            return None

        result = image.crop((min_x, min_y, max_x, max_y))
        return pad_to_max_ratio(result)
    
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
    def _recognize_batch(self, blocks):
        """
        Recognize several blocks of lines, one crop per block.

        Args:
            blocks: List of (crop, k) — crop is the PIL Image of the block (or None),
                    k the actual number of lines in the block (may be < lines_per_crop)

        Returns:
            One dict per block: {'texts': [{'text', 'confidence'}] * k, 'n_mismatch', 'n_deduped'}
        """
        results = [
            {'texts': [{'text': '', 'confidence': 0.0} for _ in range(k)],
             'n_mismatch': 0, 'n_deduped': 0}
            for _, k in blocks
        ]
        valid = [(i, crop) for i, (crop, _) in enumerate(blocks) if crop is not None]
        if not valid:
            return results

        if self.is_minicpm:
            raw_outputs = [
                self._generate_from_messages(self._prepare_messages([crop]))
                for _, crop in valid
            ]
        else:
            batch_messages = [self._prepare_messages([crop]) for _, crop in valid]
            inputs = self.processor.apply_chat_template(
                batch_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                padding=True,
                enable_thinking=False,
                return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                gen_kwargs = self._build_base_gen_kwargs()
                gen_kwargs.update({"max_new_tokens": self.max_new_tokens, "do_sample": False})
                generated_ids = self.model.generate(**inputs, **gen_kwargs)

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            raw_outputs = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            del inputs, generated_ids, generated_ids_trimmed

        for (i, _), raw in zip(valid, raw_outputs):
            k = blocks[i][1]
            texts, n_mismatch, n_deduped = parse_multiline_output(raw, k, self.dedup_threshold)
            if n_mismatch:
                n_got = len(raw.split('\n'))
                print(f"  [mismatch] block of {k} lines: got {n_got} raw lines "
                      f"({n_deduped} deduped); raw={raw[:300]!r}")
            results[i] = {
                'texts': [{'text': t, 'confidence': 1.0} for t in texts],
                'n_mismatch': n_mismatch,
                'n_deduped': n_deduped,
            }
        return results

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

                image_path, lines, _ = read_lines_geometry(alto_path)
                
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
                
                # One bounding-box crop per block of k lines within a zone
                lineblocks = group_lines_into_blocks(lines, self.lines_per_crop)
                blocks = [
                    (self._extract_multiline_image(page_image, [line.get('boundary') for line in block]),
                     len(block))
                    for block in lineblocks
                ]

                # Process in batches
                block_results = []
                for i in range(0, len(blocks), self.batch_size):
                    block_results.extend(self._recognize_batch(blocks[i:i+self.batch_size]))

                recognized_texts = [t for r in block_results for t in r['texts']]
                # Each block result holds exactly len(block) texts, so the flattened list
                # stays aligned with the flattened blocks, i.e. with `lines`.
                recognized_texts_by_id = {
                    line['id']: result
                    for line, result in zip((ln for block in lineblocks for ln in block), recognized_texts)
                }
                n_mismatch = sum(r['n_mismatch'] for r in block_results)
                n_deduped = sum(r['n_deduped'] for r in block_results)

                # Save to ALTO
                output_path = os.path.join(output_dir, os.path.basename(alto_path))
                
                if not os.path.exists(output_path):
                    copy_alto_without_text(alto_path, output_path)
                
                write_text_to_alto(output_path, recognized_texts_by_id, output_path)
                
                results.append({
                    'file': alto_path,
                    'texts': recognized_texts,
                    'n_blocks': len(blocks),
                    'n_mismatch': n_mismatch,
                    'n_deduped': n_deduped,
                })
                
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


    def _prepare_training_data_lines(self, data_path):
        """
        Prepare multi-line training samples from ALTO XML + page images.
        One sample per block of at most k consecutive lines of a TextBlock
        (group_lines_into_blocks). Crops are performed lazily in _format_conversation.

        A line with empty GT text or without polygon also closes the current block:
        it would be visible in the bounding-box crop (or missing from it) while absent
        from the target, which would teach the model to skip or invent lines.

        Pages with fewer than k usable lines (e.g. line-level datasets with one line per
        ALTO file) are skipped for now.

        Returns:
            List of dicts with page_image_path, text (list of k strings), boundary (list of k polygons)
        """

        samples = []
        xml_files = glob.glob(os.path.join(str(data_path), "**", "*.xml"), recursive=True)
        skipped = 0
        skipped_lines = 0
        skipped_short = 0

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

            _, lines, _ = read_lines_geometry(xml_path)

            # Split each TextBlock into runs of usable lines; the run index acts as a zone
            # boundary for group_lines_into_blocks.
            usable = []
            run = 0
            for line in lines:
                if not line.get('text', '').strip() or not line.get('boundary'):
                    skipped_lines += 1
                    run += 1
                    continue
                usable.append({**line, 'block_id': (line.get('block_id'), run)})

            if len(usable) < self.lines_per_crop:
                skipped_short += 1
                continue

            for block in group_lines_into_blocks(usable, self.lines_per_crop):
                samples.append({
                    "page_image_path": image_path,
                    "text": [line['text'] for line in block],
                    "boundary": [line['boundary'] for line in block],
                })

        if skipped:
            print(f"  Warning: {skipped} XML files had no matching image")
        if skipped_short:
            print(f"  Skipped {skipped_short} pages with fewer than {self.lines_per_crop} usable lines")
        if skipped_lines:
            print(f"  Skipped {skipped_lines} lines with empty GT or no polygon (block boundaries)")
        print(f"  Extracted {len(samples)} blocks of <= {self.lines_per_crop} lines "
              f"from {len(xml_files) - skipped - skipped_short} pages")
        return samples

    def _format_conversation(self, example, get_page_image):
        try:
            page_img = get_page_image(example["page_image_path"])
            img = self._extract_multiline_image(page_img, example["boundary"])
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
                # Same separator as parse_multiline_output
                "content": [{"type": "text", "text": "\n".join(example["text"])}],
            },
        ]}

    def train(self, data_path=None, seed=42):
        from unsloth import FastVisionModel
        from unsloth.trainer import UnslothVisionDataCollator
        from transformers import AutoProcessor, EarlyStoppingCallback
        from trl import SFTTrainer, SFTConfig
        """
        Fine-tune the VLM model on blocks of k lines using Unsloth.
        Each training sample is one bounding-box crop of k lines + their GT texts joined by '\n'.
        Same mechanics as VLMLineHTRTask.train(); with base_model set, the LoRA adapter in
        model_name is merged first and a new LoRA is trained on top of it.
        """
        if not data_path:
            raise ValueError("Training data path is required")

        data_paths = data_path if isinstance(data_path, list) else [data_path]
        global_path = str(data_paths[0].parent)

        print(f"Starting VLM multi-line fine-tuning with Unsloth (k={self.lines_per_crop})")
        print(f"Model: {self.model_name}")

        print("Preparing multi-line training data...")
        prompt_tpl = self.prompt_template

        train_samples = []
        for src_path in data_paths:
            train_samples.extend(self._prepare_samples_with_conventions(src_path, prompt_tpl))

        valid_samples = self._prepare_samples_with_conventions(Path(global_path) / "valid", prompt_tpl)

        if not train_samples:
            raise ValueError("No valid multi-line training samples found")

        print(f"Found {len(train_samples)} blocks (train) and {len(valid_samples)} (valid)")

        valid_train = [s for s in train_samples if os.path.exists(s["page_image_path"]) and s.get("boundary")]
        if len(valid_train) < len(train_samples):
            print(f"  Skipped {len(train_samples) - len(valid_train)} samples")

        alpha = self.hyperparams.get('special_char_weighting') or 0
        if alpha != 0:
            densities = [special_char_density("\n".join(s["text"])) for s in valid_train]
            sample_weights = [1.0 + alpha * d for d in densities]
            rng = torch.Generator()
            rng.manual_seed(seed)
            weighted_indices = torch.multinomial(
                torch.tensor(sample_weights, dtype=torch.float32),
                num_samples=len(valid_train),
                replacement=True,
                generator=rng,
            ).tolist()
            valid_train = [valid_train[i] for i in weighted_indices]
            print(f"✓ Density-weighted resampling: {len(valid_train)} samples")

        converted_train_set = _LazyLineDataset(valid_train, self._format_conversation)

        if valid_samples:
            valid_valid = [s for s in valid_samples if os.path.exists(s["page_image_path"]) and s.get("boundary")]
            converted_valid_set = _LazyLineDataset(valid_valid, self._format_conversation)
        else:
            converted_valid_set = None

        print("Loading model with Unsloth...")
        model, tokenizer = FastVisionModel.from_pretrained(
            self.model_name,
            load_in_4bit=self.hyperparams['use_4bit'],
            load_in_8bit=self.hyperparams['use_8bit'],
            use_gradient_checkpointing="unsloth",
        )
        if self.hyperparams.get('base_model'):
            # Same continuation mechanism as VLMLineHTRTask.train(): model_name is a LoRA
            # adapter whose adapter_config points to its base; merge it before the new LoRA.
            merged_dir = self.hyperparams['model_dir'] + '/merged_base'
            print(f"Merging LoRA from {self.model_name}...")
            model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
            print(f"Merged model saved to {merged_dir}")
            model = model.merge_and_unload()
            print("Merge done.")

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
            save_steps=self.hyperparams['save_steps'],
            logging_steps=self.hyperparams['logging_steps'],
            eval_strategy="steps" if valid_samples else "no",
            eval_steps=self.hyperparams['eval_steps'],
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

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=converted_train_set,
            eval_dataset=converted_valid_set,
            data_collator=UnslothVisionDataCollator(model, self.processor),
        )
        trainer.add_callback(EarlyStoppingCallback(
            early_stopping_patience=200,
            early_stopping_threshold=0.01,
        ))

        print("Starting training...")
        checkpoint_dir = self.hyperparams['model_dir']
        has_checkpoint = any(
            Path(checkpoint_dir, d).is_dir()
            for d in os.listdir(checkpoint_dir)
            if d.startswith("checkpoint-")
        ) if os.path.exists(checkpoint_dir) else False

        trainer.train(resume_from_checkpoint=has_checkpoint if has_checkpoint else None)

        model_save_path = f"{training_args.output_dir}/{self.model_name.rstrip('/').split('/')[-1]}-multiline-k{self.lines_per_crop}"
        print(f"Saving fine-tuned model to {model_save_path}")
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"Training complete! Model saved to {model_save_path}")

        del model, tokenizer, trainer
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
