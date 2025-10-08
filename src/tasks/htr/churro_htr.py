from src.tasks.htr.base_htr import BaseHTR
import os
import glob
import subprocess
from tqdm import tqdm
from pathlib import Path
import shutil
import tempfile


class ChurroHTRTask(BaseHTR):
    """
    HTR implementation using CHURRO VLM.
    Processes images directly without requiring line segmentation.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "HTR (CHURRO)"
        
        # CHURRO-specific config
        self.churro_path = config.get('churro_path', './churro')
        self.max_concurrency = config.get('max_concurrency', 8)
        self.resize = config.get('resize', None)
    
    def load(self):
        """
        Validate CHURRO installation.
        """
        self.churro_path = os.path.abspath(self.churro_path)

        if not os.path.exists(self.churro_path):
            raise FileNotFoundError(
                f"CHURRO repository not found at {self.churro_path}. "
                "Clone it: git clone https://github.com/stanford-oval/churro.git"
            )
        
        churro_script = os.path.join(self.churro_path, "run_churro_ocr.py")
        if not os.path.exists(churro_script):
            raise FileNotFoundError(f"CHURRO script not found: {churro_script}")
        
        print("CHURRO setup validated.")
    
    def _run_churro_inference(self, image_dir, output_dir):
        """
        Run CHURRO inference on a directory of images.
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save text outputs
            
        Returns:
            Dictionary mapping image filenames to recognized text
        """        
        cmd = [
            "python", "run_churro_ocr.py",
            "--engine", "churro",
            "--image-dir", str(os.path.abspath(image_dir)),
            "--pattern", "*.jpg",
            "--output-dir", str(output_dir),
            "--max-concurrency", str(self.max_concurrency)
        ]
        
        if self.resize:
            cmd.extend(["--resize", str(self.resize)])
        
        print(f"Running CHURRO: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.churro_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            print(result.stdout)
            
            # Parse outputs (CHURRO saves .txt files)
            results = {}
            for txt_file in glob.glob(os.path.join(output_dir, "*.txt")):
                base_name = Path(txt_file).stem
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                results[base_name] = text
            
            return results
            
        except subprocess.CalledProcessError as e:
            print(f"Error running CHURRO: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            raise
    
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
        # Find images
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(data_path, ext)))
        
        if not image_paths:
            raise ValueError(f"No images found in {data_path}")
        
        print(f"Found {len(image_paths)} images")
        
        # Run CHURRO in temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            print("Running CHURRO inference...")
            text_results = self._run_churro_inference(data_path, temp_dir)
            
            # Convert to ALTO format
            print("Converting to ALTO format...")
            results = []
            for image_path in tqdm(image_paths, desc="Creating ALTO files"):
                base_name = Path(image_path).stem
                text = text_results.get(base_name, '')
                
                output_path = os.path.join(output_dir, f"{base_name}.xml")
                self._create_simple_alto_with_text(image_path, text, output_path)
                
                results.append({
                    'file': image_path,
                    'text': text
                })
                
                if save_image:
                    image_output = os.path.join(output_dir, os.path.basename(image_path))
                    if not os.path.exists(image_output):
                        shutil.copy2(image_path, image_output)
        
        return results