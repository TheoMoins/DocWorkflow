from src.tasks.htr.vlm_line_htr import VLMLineHTRTask
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

from transformers import TrainerCallback

from src.content.weighted_sampling import special_char_density
from src.alto.alto_lines import extract_lines_from_alto
from src.alto.alto_text import copy_alto_without_text

Image.MAX_IMAGE_PIXELS = None



class VLMLineHTRTaskSilver(VLMLineHTRTask):
    """
    For performing continued pre-training of the language heads 
    of VLMs on additional textual data (for learning new languages 
    or domains). Training data should consist of a folder of text files.
    Each line from each text file is treated as a sample.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "HTR_VLM_Line_Level_Silver"
    
        

    def _prepare_training_data_textonly(self, data_path):
        """
        Prepare line-level training samples from plain text files
        Splits on lines in text file
        If line is too long (such as when text file does not include line breaks)
        splits every n characters to produce lines of reasonable length
        Returns:
            dataset
        """
        from datasets import load_dataset, Dataset
        import pandas as pd
        from textwrap import wrap

        n = 1024 # max line length in characters
        
        ds = load_dataset("text", data_dir=data_path, split='train')
        df = ds.to_pandas()
        df['text'] = df['text'].astype(object)
        df['len'] = (df['text'].astype(str).str.len())
        df = df[df['len']>0].reset_index(drop=True)
        df.loc[df['len']>n,'text'] = df['text'][df['len']>n].apply(lambda x: wrap(x, width=n, expand_tabs=False,replace_whitespace=False, break_on_hyphens=False, break_long_words = False))
        dataset = Dataset.from_pandas(df.explode('text').drop(columns=['len']).reset_index(drop=True))
        return dataset
        
    def train(self, data_path=None, seed=42):
        import unsloth
        from unsloth import FastVisionModel, FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
        from unsloth.trainer import UnslothVisionDataCollator
        from transformers import AutoProcessor, DataCollatorForLanguageModeling, TrainingArguments
        from trl import SFTTrainer, SFTConfig
        """
        Perform continued pre-training of language head of VLM on text data.
        Each training sample is a line of text.
        """
        print("To train this model, you must change the environment to vlm-training:")
        print("\n  source envs/vlm-training/bin/activate")

        if not data_path:
            raise ValueError("Training data path is required")

        data_paths = data_path if isinstance(data_path, list) else [data_path]

        global_path = str(data_paths[0].parent)

        print(f"Starting VLM continued pre-training with Unsloth")
        print(f"Model: {self.model_name}")

        print("Preparing line-level training data...")
        train_samples = self._prepare_training_data_textonly(data_path)
        #TODO: We are currently not using the validation samples I think    
        valid_samples = self._prepare_training_data_lines(global_path + "/valid")

        if not train_samples:
            raise ValueError("No valid line-level training samples found")

        print(f"Found {len(train_samples)} line samples (train) and {len(valid_samples)} (valid)")


        print("Loading model with Unsloth...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            self.model_name,
            load_in_4bit=self.hyperparams['use_4bit'],
            load_in_8bit=self.hyperparams['use_8bit'],
            use_gradient_checkpointing="unsloth",
        )

        model = FastLanguageModel.get_peft_model(
            model,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],# remove lm_head as embeddings are tied?, "lm_head",], # Add for continual pretraining
            modules_to_save =["embed_tokens"], # move to modules to save only?
            r=self.hyperparams['lora_r'],
            lora_alpha=self.hyperparams['lora_r']/2,
            lora_dropout=self.hyperparams['lora_dropout'],
            use_rslora=self.hyperparams['use_rslora'],
            loftq_config=None
        )


        EOS_TOKEN = tokenizer.eos_token
        def formatting_prompts_func(examples):
            return { "text" : [example + EOS_TOKEN for example in examples["text"]] }
        train_samples = train_samples.map(formatting_prompts_func, batched = True,)
        # This is too long to process!
        # cer_callback = CEREvalCallback(
        #     model=model,
        #     processor=self.processor,
        #     eval_samples=list(converted_valid_set)[:50] if converted_valid_set else [],
        #     device=self.device,
        # )

        training_args = UnslothTrainingArguments(
                per_device_train_batch_size = self.hyperparams['train_batch_size'],
                gradient_accumulation_steps = self.hyperparams['gradient_accumulation_steps'],

                # Use warmup_ratio and num_train_epochs for longer runs!
                # max_steps = 10, #120
                # warmup_steps = 1, #10
                warmup_ratio=self.hyperparams['warmup_ratio'],
                num_train_epochs=self.hyperparams['epochs'],

                # Select a 2 to 10x smaller learning rate for the embedding matrices!
                learning_rate = self.hyperparams['learning_rate'],#5e-5,
                embedding_learning_rate = self.hyperparams['learning_rate']/5,

                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay=self.hyperparams['weight_decay'],
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = "outputs",
                report_to = "none", # Use TrackIO/WandB etc
                remove_unused_columns = True, # Fix for ValueError
            )

        trainer = UnslothTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = train_samples,
            dataset_text_field = "text",
            max_seq_length = self.hyperparams['max_seq_length'],
            dataset_num_proc = 4,

            #TODO make these args from settings
            args = training_args,
        )


        #trainer.add_callback(early_stopping_callback) #AssertionError: EarlyStoppingCallback requires IntervalStrategy of steps or epoch

        print("Starting training...")
        # remove checkpointing for now
        '''checkpoint_dir = self.hyperparams['output_dir']
        has_checkpoint = any(
            Path(checkpoint_dir, d).is_dir()
            for d in os.listdir(checkpoint_dir)
            if d.startswith("checkpoint-")
        ) if os.path.exists(checkpoint_dir) else False'''

        trainer.train()#resume_from_checkpoint=has_checkpoint if has_checkpoint else None)

        model_save_path = f"{training_args.output_dir}/{self.model_name.split('/')[-1]}-{self.name}"
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
