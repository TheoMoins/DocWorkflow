from src.tasks.htr.base_vlm_htr import BaseVLMHTR
import os
import shutil
import gc
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from lxml import etree as ET

from src.alto.alto_lines import extract_lines_from_alto


class VLMLineHTRTask(BaseVLMHTR):
    """
    HTR using VLM for line-level transcription.
    Processes pre-segmented lines from ALTO XML files.
    Examples: Qwen3-VL-2B-catmus, Idefics3, etc.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "HTR (VLM Line-Level)"
        self.batch_size = config.get('line_batch_size', 1)
    
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
                    shutil.copy2(alto_path, output_path)
                
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