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
    Extract text line by line from ALTO XML file.
    
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
        
        strings = textline.findall('.//alto:String', ns)
        if strings:
            text = ' '.join([s.get('CONTENT', '') for s in strings if s.get('CONTENT')])
        else:
            text = ''
        
        lines_text.append({
            'id': line_id,
            'text': text
        })
    
    return lines_text