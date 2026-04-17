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

from src.alto.alto_text import copy_and_fix_alto_namespaces, read_document_text, create_minimal_alto, split_text_into_alto_lines

Image.MAX_IMAGE_PIXELS = None


class VLMPageHTRTask(BaseVLMHTR):
    """
    HTR using VLM for page-level transcription.
    Processes entire page images and splits output into lines.
    Examples: CHURRO, etc.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "HTR_VLM_Page_Level"
    
    

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
                    create_minimal_alto(image_path, text, output_path)
                
                # Split VLM output into lines
                split_text_into_alto_lines(output_path, text, image_path)
                
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
            load_in_8bit=self.hyperparams['use_8bit'],
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
                use_fast=False,
                max_pixels=self.hyperparams['max_pixels']
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
        
        config_path = self._create_finetuned_config(model_save_path, global_path, 'VLMPAGEHTR')
        
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
            text = read_document_text(xml_path)
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
