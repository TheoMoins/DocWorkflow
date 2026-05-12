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
from src.tasks.htr.prompt_convention import build_conventions_block, load_conventions


from src.alto.alto_text import copy_and_fix_alto_namespaces, read_fullpage_cleaned, create_minimal_alto, split_text_into_alto_lines

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
    
    def _get_file_extensions(self):
        """VLM Line-Level works with ALTO XML files."""
        return ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']

    def _recognize_single_image(self, image):
        messages = self._prepare_messages(Image.open(image).convert("RGB"))
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
                #Load conventions
                doc_dir = Path(image_path).parent
                conventions = load_conventions(doc_dir)
                prompt_tpl = self.prompt_template
                if conventions and '{conventions}' in prompt_tpl:
                    self.prompt = prompt_tpl.replace('{conventions}', build_conventions_block(conventions))
                else:
                    self.prompt = prompt_tpl.replace('{conventions}', '').strip()


                # Recognize text
                text = self._recognize_single_image(image_path)
                
                # Create basic ALTO file
                base_name = Path(image_path).stem
                output_path = os.path.join(output_dir, f"{base_name}.xml")
                
                ## Check if there's an existing ALTO with layout/lines
                #existing_alto = os.path.join(source_dir, f"{base_name}.xml")
                #if os.path.exists(existing_alto):
                #    # Copy existing structure AND clean namespaces
                #    copy_and_fix_alto_namespaces(existing_alto, output_path)
                #else:
                #    # Create simple ALTO
                #    create_minimal_alto(image_path, text, output_path)
                
                ## Split VLM output into lines
                #split_text_into_alto_lines(output_path, text, image_path)
                
                # we are hoping the output text is a valid xml!!
                with open(output_path, "w") as f:
                    f.write(text)

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

    #TODO: remove redundancy between this and line-level
    '''
    def format_conversation(self, example):
        img = Image.open(example["image_path"]).convert("RGB")
        prompt = example.get("prompt", self.prompt)
        if img is None:
            return None
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
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
        
        return {"messages": conversation}    '''

    def _format_conversation(self, example, get_page_image):
        try:
            img = get_page_image(example["image_path"])
        except Exception as e:
            print(f"Warning: skipping sample ({example.get('image_path', '?')}): {e}")
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

    def _prepare_training_data(self, data_path):
        """
        Prepare training data with conversation structure.
        Returns a list of image paths and texts, images will be loaded on-the-fly.
        """
        samples = []
        xml_files = glob.glob(os.path.join(data_path, "*.xml"))
        
        for xml_path in xml_files:
            # Extract text
            # TODO: modify document text function to get
            # stripped & sorted alto file text
            text = read_fullpage_cleaned(xml_path)
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

    # should this also check if text is valid?
    def _validate_samples(self, examples):
        return [s for s in examples if os.path.exists(s["image_path"])]
    '''
    def _convert_set(self, examples):
        return [self.format_conversation(sample) for sample in examples] '''
