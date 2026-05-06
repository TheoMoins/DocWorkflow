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
from src.alto import ALTO_NS, ALTO_NS_PREFIX
from src.alto.alto_lines import read_lines_geometry
from src.alto.alto_text import copy_alto_without_text, write_text_to_alto
from src.utils.lazy_dataset import LazyLineDataset

Image.MAX_IMAGE_PIXELS = None

class VLMLineHTRTask(BaseVLMHTR):
    """
    HTR using VLM for line-level transcription.
    Processes pre-segmented lines from ALTO XML files.
    Examples: Qwen3-VL-2B-catmus, Idefics3, etc.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "HTR_VLM_Line_Level"
    

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
                
                # Extract all line images
                line_images = []
                line_ids = []
                for line in lines:
                    line_ids.append(line['id'])
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
                
                recognized_texts_by_id = {
                    line_id: result
                    for line_id, result in zip(line_ids, recognized_texts)
                }
                
                # Save to ALTO
                output_path = os.path.join(output_dir, os.path.basename(alto_path))
                
                if not os.path.exists(output_path):
                    copy_alto_without_text(alto_path, output_path)
                
                write_text_to_alto(output_path, recognized_texts_by_id, output_path)
                
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
    

    def _prepare_training_data(self, data_path):
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

            skipped_empty_gt = 0
            _, lines, _ = read_lines_geometry(xml_path)
            for line in lines:
                if not line.get('text', '').strip():
                    skipped_empty_gt += 1
                    continue
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
    
    def _validate_samples(self, examples):
        return [s for s in examples if os.path.exists(s["page_image_path"]) and s.get("boundary")]
    
