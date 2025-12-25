import glob
import pandas as pd
from pathlib import Path
from lxml import etree as ET
import jiwer
import os

def extract_text_from_alto(alto_path):
    """
    Extract transcribed text from ALTO XML file.
    
    Args:
        alto_path: Path to ALTO XML file
        
    Returns:
        String containing the full transcription
    """
    tree = ET.parse(alto_path)
    root = tree.getroot()
    ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
    
    # Extract all String elements with CONTENT
    strings = root.findall('.//alto:String', ns)
    texts = [s.get('CONTENT', '') for s in strings if s.get('CONTENT')]
    
    # Join with spaces to preserve word boundaries
    return ' '.join(texts) if texts else ''

def create_solution_csv(data_dir, output_csv):
    """
    Create solution CSV from ALTO files.
    
    Args:
        data_dir: Directory containing ALTO XML files
        output_csv: Path to save the solution CSV
    """
    xml_files = sorted(glob.glob(os.path.join(data_dir, "*.xml")))
    
    if not xml_files:
        raise ValueError(f"No XML files found in {data_dir}")
    
    data = []
    empty_files = []
    
    for xml_path in xml_files:
        image_id = Path(xml_path).stem
        text = extract_text_from_alto(xml_path)
        
        if not text.strip():
            empty_files.append(image_id)
            continue
        
        data.append({
            'image_id': image_id,
            'transcription': text
        })
    
    if empty_files:
        print(f"⚠️  Warning: {len(empty_files)} files with empty transcriptions:")
        for f in empty_files[:5]:
            print(f"  - {f}")
        if len(empty_files) > 5:
            print(f"  ... and {len(empty_files) - 5} more")
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Created solution CSV: {output_csv}")
    print(f"  - {len(df)} valid transcriptions")
    print(f"  - Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())

def create_submission_template(data_dir, output_csv):
    """
    Create empty submission template from images.
    
    Args:
        data_dir: Directory containing images
        output_csv: Path to save the template CSV
    """
    # Find all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
    images = []
    
    for ext in image_extensions:
        images.extend(glob.glob(os.path.join(data_dir, ext)))
    
    if not images:
        raise ValueError(f"No images found in {data_dir}")
    
    images = sorted(images)
    
    # Create template
    data = []
    for img_path in images:
        image_id = Path(img_path).stem
        data.append({
            'image_id': image_id,
            'transcription': ''  # To be filled
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    
    print(f"✓ Created submission template: {output_csv}")
    print(f"  - {len(df)} images")


if __name__ == "__main__":
    # Créer le fichier solution (secret)
    create_solution_csv("results/catmus_pipeline_htromance/htr", "sandbox_submission.csv")
    
