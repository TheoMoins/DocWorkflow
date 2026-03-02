from lxml import etree as ET


def extract_text_from_alto(alto_path):
    """
    Extract transcribed text from ALTO XML file.
    
    Args:
        alto_path: Path to ALTO XML file
        
    Returns:
        Concatenated text from all String elements
    """
    tree = ET.parse(alto_path)
    root = tree.getroot()
    ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
    
    strings = root.findall('.//alto:String', ns)
    texts = [s.get('CONTENT', '') for s in strings if s.get('CONTENT')]
    
    return ' '.join(texts) if texts else ''


def extract_lines_text_from_alto(alto_path):
    """
    Extract text line by line from ALTO XML file, sorted in reading order.
    
    Args:
        alto_path: Path to ALTO XML file
        
    Returns:
        List of dictionaries with line_id and text content
    """
    tree = ET.parse(alto_path)
    root = tree.getroot()
    ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
    
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
    NSMAP = {
        None: "http://www.loc.gov/standards/alto/ns-v4#",
        "xsi": "http://www.w3.org/2001/XMLSchema-instance"
    }
    
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

def extract_lines_dict_from_alto(alto_path):
    """
    Extract text from ALTO XML as a dict of line_id -> text,
    matching the CMMHWR competition submission format.

    Args:
        alto_path: Path to ALTO XML file

    Returns:
        dict of line_id -> text content
    """
    tree = ET.parse(alto_path)
    root = tree.getroot()
    ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}

    lines = {}
    for textline in root.findall('.//alto:TextLine', ns):
        line_id = textline.get('ID', '')
        if not line_id:
            continue
        strings = textline.findall('.//alto:String', ns)
        text = ' '.join(s.get('CONTENT', '') for s in strings if s.get('CONTENT'))
        lines[line_id] = text

    return lines

def extract_lines_with_bbox_from_alto(alto_path):
    """
    Extract text lines with bounding box coordinates from ALTO XML.
    
    Returns:
        List of dicts with id, text, hpos, vpos, width, height
    """
    tree = ET.parse(alto_path)
    root = tree.getroot()
    ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
    
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
    ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
    for string_elem in root.findall('.//alto:String', ns):
        string_elem.set('CONTENT', '')
        string_elem.set('WC', '0.0')
    tree.write(dst_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")