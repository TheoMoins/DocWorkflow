"""
Conversion .parquet → ALTO (une image + un XML par ligne)
Pour évaluation HTR avec lignes pré-segmentées
"""

from datasets import load_dataset
from lxml import etree as ET
import os
from pathlib import Path
from tqdm import tqdm
import hashlib
import click


def create_single_line_alto(row, output_dir, line_idx):
    """
    Crée une image + un fichier ALTO contenant UNE seule ligne.
    """
    
    NSMAP = {
        None: "http://www.loc.gov/standards/alto/ns-v4#",
        "xsi": "http://www.w3.org/2001/XMLSchema-instance"
    }
    
    # Extraire l'image de la ligne
    line_img = row['im']
    width, height = line_img.size
    
    # Créer un nom de fichier unique
    shelfmark = str(row.get('shelfmark', 'unknown'))
    safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' 
                       for c in shelfmark)
    safe_name = safe_name.replace(' ', '_')
    
    # Nom unique avec index
    base_filename = f"{safe_name}_line_{line_idx:05d}"
    image_filename = f"{base_filename}.jpg"
    
    # Sauvegarder l'image
    image_path = os.path.join(output_dir, image_filename)
    line_img.save(image_path, 'JPEG', quality=95)
    
    # Créer l'ALTO
    alto = ET.Element("alto", nsmap=NSMAP, attrib={
        "{http://www.w3.org/2001/XMLSchema-instance}schemaLocation":
            "http://www.loc.gov/standards/alto/ns-v4# http://www.loc.gov/standards/alto/v4/alto-4-2.xsd"
    })
    
    # Description
    description = ET.SubElement(alto, "Description")
    ET.SubElement(description, "MeasurementUnit").text = "pixel"
    source_info = ET.SubElement(description, "sourceImageInformation")
    ET.SubElement(source_info, "fileName").text = image_filename
    
    # Tags
    tags_section = ET.SubElement(alto, "Tags")
    
    region_type = row.get('region', 'MainZone')
    tag_id = f"BT{hashlib.sha1(str(region_type).encode()).hexdigest()[:6]}"
    ET.SubElement(tags_section, "OtherTag", ID=tag_id, LABEL=str(region_type), 
                 DESCRIPTION=f"block type {region_type}")
    
    line_type = row.get('line_type', 'DefaultLine')
    line_tag_id = f"LT{hashlib.sha1(str(line_type).encode()).hexdigest()[:6]}"
    ET.SubElement(tags_section, "OtherTag", ID=line_tag_id, LABEL=str(line_type),
                 DESCRIPTION=f"line type {line_type}")
    
    # Layout
    layout = ET.SubElement(alto, "Layout")
    page = ET.SubElement(layout, "Page", ID="page1", PHYSICAL_IMG_NR="1",
                        HEIGHT=str(height), WIDTH=str(width))
    print_space = ET.SubElement(page, "PrintSpace", 
                               HEIGHT=str(height), WIDTH=str(width),
                               VPOS="0", HPOS="0")
    
    # TextBlock unique (toute l'image)
    text_block = ET.SubElement(print_space, "TextBlock", ID="block_0",
                              HPOS="0", VPOS="0",
                              WIDTH=str(width), HEIGHT=str(height),
                              TAGREFS=tag_id)
    
    margin = 2  # Marge de sécurité pour éviter les dépassements

    text_line = ET.SubElement(text_block, "TextLine", ID="line_0",
                            HPOS=str(margin), 
                            VPOS=str(margin),
                            WIDTH=str(width - 2 * margin), 
                            HEIGHT=str(height - 2 * margin),
                            TAGREFS=line_tag_id)

    baseline_y = height // 2
    text_line.set('BASELINE', f"{margin} {baseline_y} {width - margin} {baseline_y}")

    shape = ET.SubElement(text_line, "Shape")
    points = f"{margin} {margin} {width - margin} {margin} {width - margin} {height - margin} {margin} {height - margin}"
    ET.SubElement(shape, "Polygon", POINTS=points)
    
    # Ground truth
    text = str(row.get('text', ''))
    if text:
        string_elem = ET.SubElement(text_line, "String")
        string_elem.set('CONTENT', text)
        string_elem.set('WC', '1.0')
    
    # Sauvegarder l'ALTO
    xml_filename = f"{base_filename}.xml"
    output_path = os.path.join(output_dir, xml_filename)
    
    tree = ET.ElementTree(alto)
    tree.write(output_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    
    return output_path


@click.command()
@click.option("-i", "--input", required=True, type=click.Path(exists=True))
@click.option("-o", "--output", required=True, type=click.Path())
def main(input, output):
    """
    Convert .parquet to individual line images + ALTO XML files.
    Each line becomes one .jpg + one .xml file.
    
    Example:
        python convert_lines_individually.py -i data/catmus/ -o data/catmus_lines/
    """
    
    input_path = Path(input)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"LINE-BY-LINE CONVERTER")
    print(f"{'='*60}\n")
    
    parquet_files = list(input_path.glob("*.parquet"))
    
    if not parquet_files:
        print(f"❌ No .parquet files found in {input_path}")
        return
    
    print(f"Found {len(parquet_files)} .parquet file(s)")
    
    total_lines = 0
    
    for parquet_file in parquet_files:
        print(f"\nProcessing: {parquet_file.name}")
        
        try:
            # Charger avec datasets pour décoder les images
            dataset = load_dataset('parquet', data_files=str(parquet_file), split='train')
            print(f"  Total lines: {len(dataset)}")
            
            # Convertir chaque ligne individuellement
            for idx, row in enumerate(tqdm(dataset, desc="  Converting", unit="line")):
                # PRENDRE UNIQUEMENT LES CINQ PREMIERES LIGNES DE CHAQUE FICHIER
                if idx > 5:
                    break
                try:
                    create_single_line_alto(row, str(output_path), total_lines + idx)
                except Exception as e:
                    print(f"\n  ❌ Error on line {idx}: {e}")
            
            total_lines += len(dataset)
            
        except Exception as e:
            print(f"  ❌ Error loading {parquet_file}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"✅ CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"Total lines: {total_lines}")
    print(f"Output: {output_path}")
    print(f"\nFiles created:")
    print(f"  - {total_lines} × .jpg (line images)")
    print(f"  - {total_lines} × .xml (ALTO with ground truth)")
    print(f"\n📝 NEXT STEPS:")
    print(f"{'='*60}")
    print(f"1. Update config.yml:")
    print(f"   data:")
    print(f"     test: \"{output_path}\"")
    print(f"\n2. Run HTR evaluation:")
    print(f"   docworkflow -c config.yml predict -t htr -d test")
    print(f"   docworkflow -c config.yml score -t htr -d test")
    print(f"\n⚠️  Note: This dataset has pre-segmented lines.")
    print(f"   Layout and line tasks will be skipped/trivial.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()