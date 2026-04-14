import os
from lxml import etree as ET
from PIL import Image

from src.alto import ALTO_NS, ALTO_NS_MAP, ALTO_NS_PREFIX

def read_document_text(alto_path):
    """
    Extract transcribed text from ALTO XML file.
    
    Args:
        alto_path: Path to ALTO XML file
        
    Returns:
        Concatenated text from all String elements
    """
    tree = ET.parse(alto_path)
    root = tree.getroot()
    ns = ALTO_NS_PREFIX
    
    strings = root.findall('.//alto:String', ns)
    texts = [s.get('CONTENT', '') for s in strings if s.get('CONTENT')]
    
    return ' '.join(texts) if texts else ''


def read_lines_text(alto_path):
    """
    Extract text line by line from ALTO XML file, sorted in reading order.
    
    Args:
        alto_path: Path to ALTO XML file
        
    Returns:
        List of dictionaries with line_id and text content
    """
    tree = ET.parse(alto_path)
    root = tree.getroot()
    ns = ALTO_NS_PREFIX
    
    lines_text = []
    
    for textline in root.findall('.//alto:TextLine', ns):
        line_id = textline.get('ID', '')
        vpos = int(float(textline.get('VPOS', 0)))
        hpos = int(float(textline.get('HPOS', 0)))
        
        strings = textline.findall('.//alto:String', ns)
        if strings:
            text = ' '.join([s.get('CONTENT', '') for s in strings if s.get('CONTENT')])
        else:
            text = ''
        
        lines_text.append({
            'id': line_id,
            'text': text,
            'vpos': vpos,
            'hpos': hpos
        })
    
    # Sort by reading order: Y first (with tolerance), then X
    lines_text.sort(key=lambda x: (x['vpos'] // 10, x['hpos']))  # 10px tolerance for same row
    
    # Remove position info from output
    return [{'id': l['id'], 'text': l['text']} for l in lines_text]

def copy_and_fix_alto_namespaces(source_path, dest_path):
    """
    Copier un fichier ALTO en nettoyant les préfixes de namespace (ns0:, etc.).
    
    Args:
        source_path: Fichier ALTO source (peut avoir ns0:)
        dest_path: Fichier ALTO destination (sans ns0:)
    """
    from lxml import etree
    
    # Parse le fichier source
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(source_path, parser)
    old_root = tree.getroot()
    
    # Namespace map propre (sans préfixe pour ALTO)
    NSMAP = ALTO_NS_MAP
    
    # Créer nouveau root avec namespaces propres
    new_root = etree.Element("alto", nsmap=NSMAP)
    new_root.set("{http://www.w3.org/2001/XMLSchema-instance}schemaLocation",
                 old_root.get("{http://www.w3.org/2001/XMLSchema-instance}schemaLocation", 
                             "http://www.loc.gov/standards/alto/ns-v4# http://www.loc.gov/standards/alto/v4/alto-4-2.xsd"))
    
    # Fonction récursive pour copier les éléments sans préfixe
    def copy_element(source, target_parent):
        tag = etree.QName(source).localname
        new_elem = etree.SubElement(target_parent, tag)
        
        # Copier attributs (sans namespace declarations)
        for key, value in source.attrib.items():
            if 'xmlns' not in key:  # Skip namespace declarations
                attr_name = etree.QName(key).localname
                new_elem.set(attr_name, value)
        
        # Copier text
        if source.text:
            new_elem.text = source.text
        if source.tail:
            new_elem.tail = source.tail
        
        # Copier récursivement les enfants
        for child in source:
            copy_element(child, new_elem)
    
    # Copier tous les enfants
    for child in old_root:
        copy_element(child, new_root)
    
    # Sauvegarder
    new_tree = etree.ElementTree(new_root)
    new_tree.write(dest_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")

def read_lines_with_bbox(alto_path):
    """
    Extract text lines with bounding box coordinates from ALTO XML.
    
    Returns:
        List of dicts with id, text, hpos, vpos, width, height
    """
    tree = ET.parse(alto_path)
    root = tree.getroot()
    ns = ALTO_NS_PREFIX
    
    lines = []
    for textline in root.findall('.//alto:TextLine', ns):
        strings = textline.findall('.//alto:String', ns)
        text = ' '.join([s.get('CONTENT', '') for s in strings if s.get('CONTENT')])
        if not text.strip():
            continue
        try:
            hpos   = int(float(textline.get('HPOS', 0)))
            vpos   = int(float(textline.get('VPOS', 0)))
            width  = int(float(textline.get('WIDTH', 0)))
            height = int(float(textline.get('HEIGHT', 0)))
        except (ValueError, TypeError):
            continue
        if width <= 0 or height <= 0:
            continue
        lines.append({
            'id': textline.get('ID', ''),
            'text': text,
            'hpos': hpos,
            'vpos': vpos,
            'width': width,
            'height': height,
        })
    return lines


def copy_alto_without_text(src_path, dst_path):
    """Copy ALTO XML but strip all String CONTENT to avoid GT leakage."""
    tree = ET.parse(src_path)
    root = tree.getroot()
    ns = ALTO_NS_PREFIX
    for string_elem in root.findall('.//alto:String', ns):
        string_elem.set('CONTENT', '')
        string_elem.set('WC', '0.0')
    tree.write(dst_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")


def write_text_to_alto(alto_path: str, texts: list, output_path: str) -> None:
    """
    Write recognized text into an existing ALTO XML file.

    Args:
        alto_path: Path to the input ALTO file (with TextLine geometry)
        texts: List of dicts with 'text' and optional 'confidence' keys,
               one per TextLine in document order
        output_path: Path to save the modified ALTO
    """
    tree = ET.parse(alto_path)
    root = tree.getroot()

    text_lines = root.findall('.//alto:TextLine', ALTO_NS_PREFIX)

    for line, text_data in zip(text_lines, texts):
        if text_data and text_data.get('text'):
            for string_elem in line.findall('alto:String', ALTO_NS_PREFIX):
                line.remove(string_elem)
            string_elem = ET.SubElement(line, ET.QName(ALTO_NS, 'String'))
            string_elem.set('CONTENT', text_data['text'])
            string_elem.set('WC', str(text_data.get('confidence', 1.0)))

    tree.write(output_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")

def create_minimal_alto(image_path, text, output_path):
    """
    Create a simple ALTO XML file with recognized text.
    Creates one TextBlock and one TextLine covering the entire image.
    
    Args:
        image_path: Path to the source image
        text: Recognized text
        output_path: Where to save the ALTO XML
    """
    ns = "http://www.loc.gov/standards/alto/ns-v4#"
    NSMAP = {
        None: ns,
        "xsi": "http://www.w3.org/2001/XMLSchema-instance"
    }
    
    # Get image dimensions
    with Image.open(image_path) as img:
        width, height = img.size
    
    # Create ALTO structure
    alto = ET.Element("alto", nsmap=NSMAP, attrib={
        "{http://www.w3.org/2001/XMLSchema-instance}schemaLocation":
            f"{ns} http://www.loc.gov/standards/alto/v4/alto-4-2.xsd"
    })
    
    # Description
    description = ET.SubElement(alto, "Description")
    ET.SubElement(description, "MeasurementUnit").text = "pixel"
    source_info = ET.SubElement(description, "sourceImageInformation")
    ET.SubElement(source_info, "fileName").text = os.path.basename(image_path)
    
    # Tags
    tags = ET.SubElement(alto, "Tags")
    ET.SubElement(tags, "OtherTag", ID="BT1", LABEL="MainZone", 
                    DESCRIPTION="block type MainZone")
    ET.SubElement(tags, "OtherTag", ID="LT1", LABEL="DefaultLine",
                    DESCRIPTION="line type DefaultLine")
    
    # Layout
    layout = ET.SubElement(alto, "Layout")
    page = ET.SubElement(layout, "Page", ID="page1", PHYSICAL_IMG_NR="1",
                        HEIGHT=str(height), WIDTH=str(width))
    print_space = ET.SubElement(page, "PrintSpace", 
                                HEIGHT=str(height), WIDTH=str(width),
                                VPOS="0", HPOS="0")
    
    # Single TextBlock covering the whole image
    text_block = ET.SubElement(print_space, "TextBlock", ID="block_0",
                                HPOS="0", VPOS="0",
                                WIDTH=str(width), HEIGHT=str(height),
                                TAGREFS="BT1")
    
    # Single TextLine with the recognized text
    margin = 5
    text_line = ET.SubElement(text_block, "TextLine", ID="line_0",
                            HPOS=str(margin), VPOS=str(margin),
                            WIDTH=str(width - 2 * margin), 
                            HEIGHT=str(height - 2 * margin),
                            TAGREFS="LT1")
    
    # Baseline
    baseline_y = height // 2
    text_line.set('BASELINE', f"{margin} {baseline_y} {width - margin} {baseline_y}")
    
    # Shape
    shape = ET.SubElement(text_line, "Shape")
    points = f"{margin} {margin} {width - margin} {margin} {width - margin} {height - margin} {margin} {height - margin}"
    ET.SubElement(shape, "Polygon", POINTS=points)
    
    # String with recognized text
    if text:
        string_elem = ET.SubElement(text_line, "String")
        string_elem.set('CONTENT', text)
        string_elem.set('WC', '1.0')
    
    # Save
    tree = ET.ElementTree(alto)
    tree.write(output_path, pretty_print=True, 
                xml_declaration=True, encoding="UTF-8")
    
def deduplicate_alto_consecutive_lines(alto_path):
    """
    Remove consecutive duplicate TextLines from an ALTO XML file in-place.
    Targets hallucination artifacts where HTR models repeat the same line.
    """
    tree = ET.parse(alto_path)
    root = tree.getroot()
    ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}

    for text_block in root.findall('.//alto:TextBlock', ns):
        lines = text_block.findall('alto:TextLine', ns)
        prev_text = None
        for line in lines:
            strings = line.findall('.//alto:String', ns)
            text = ' '.join(s.get('CONTENT', '') for s in strings).strip()
            if text and text == prev_text:
                text_block.remove(line)
            else:
                prev_text = text

    tree.write(alto_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")

def split_text_into_alto_lines(alto_path, text, image_path):
    """Split VLM output into TextLines (same as before)."""
    from PIL import Image
    
    lines_text = [line.strip() for line in text.split('\n') if line.strip()]
    
    if not lines_text:
        return alto_path
    
    tree = ET.parse(alto_path)
    root = tree.getroot()
    ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
    
    with Image.open(image_path) as img:
        width, height = img.size
    
    existing_lines = root.findall('.//alto:TextLine', ns)
    
    if existing_lines and len(existing_lines) == len(lines_text):
        # Perfect match
        for line_elem, line_text in zip(existing_lines, lines_text):
            for string_elem in line_elem.findall('alto:String', ns):
                line_elem.remove(string_elem)
            
            string_elem = ET.SubElement(line_elem, ET.QName(ns['alto'], 'String'))
            string_elem.set('CONTENT', line_text)
            string_elem.set('WC', '1.0')
    else:
        # Create new structure
        text_blocks = root.findall('.//alto:TextBlock', ns)
        
        if text_blocks:
            text_block = text_blocks[0]
            for extra_block in text_blocks[1:]:
                extra_block.getparent().remove(extra_block)
            
            for line_elem in text_block.findall('alto:TextLine', ns):
                text_block.remove(line_elem)
            
            line_height = height // max(len(lines_text), 1)
            margin = 10
            
            for idx, line_text in enumerate(lines_text):
                y_pos = idx * line_height + margin
                line_elem = ET.SubElement(text_block, ET.QName(ns['alto'], 'TextLine'))
                line_elem.set('ID', f'line_{idx}')
                line_elem.set('HPOS', str(margin))
                line_elem.set('VPOS', str(y_pos))
                line_elem.set('WIDTH', str(width - 2 * margin))
                line_elem.set('HEIGHT', str(line_height - margin))
                
                baseline_y = y_pos + line_height // 2
                line_elem.set('BASELINE', f"{margin} {baseline_y} {width - margin} {baseline_y}")
                
                string_elem = ET.SubElement(line_elem, ET.QName(ns['alto'], 'String'))
                string_elem.set('CONTENT', line_text)
                string_elem.set('WC', '1.0')
    
    def indent_xml(elem, level=0):
        i = "\n" + level*"  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                indent_xml(child, level+1)
            if not child.tail or not child.tail.strip():
                child.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i
    
    indent_xml(root)
    ET.cleanup_namespaces(root)
    tree.write(alto_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    return alto_path