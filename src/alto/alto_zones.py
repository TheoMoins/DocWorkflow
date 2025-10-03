import os
import numpy as np

from lxml import etree as ET


def convert_zones_to_boxes(zones, image_size, is_gt=True):
    """
    Convert layout zones (TextBlocks) to bounding box format
    
    Args:
        zones: List of zone dictionaries with boundaries or bbox attributes
        image_size: (width, height) of the image
        is_gt: Whether these are ground truth boxes (True) or predictions (False)
        
    Returns:
        Numpy array of boxes
    """
    width, height = image_size
    boxes = []
    
    for i, zone in enumerate(zones):
        # Get bounding box coordinates from zone
        if 'boundary' in zone and zone['boundary']:
            boundary = np.array(zone['boundary'])
            min_x, min_y = boundary.min(axis=0)
            max_x, max_y = boundary.max(axis=0)
        elif 'bbox' in zone:
            min_x, min_y, max_x, max_y = zone['bbox']
        else:
            continue
        
        # Normalize coordinates to 0-100 scale
        x1 = int(100 * min_x / width)
        y1 = int(100 * min_y / height)
        x2 = int(100 * max_x / width)
        y2 = int(100 * max_y / height)
        
        # Get class_id from zone label if available, otherwise use 0
        class_id = zone.get('class_id', 0)
        
        if is_gt:
            # For ground truth: [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
            boxes.append([x1, y1, x2, y2, class_id, 0, 0])
        else:
            # For predictions: [xmin, ymin, xmax, ymax, class_id, confidence_score]
            confidence = zone.get('confidence', 1.0)
            boxes.append([x1, y1, x2, y2, class_id, confidence])
    
    return np.array(boxes) if boxes else np.zeros((0, 7 if is_gt else 6))


def extract_zones_from_alto(file_path):
    """
    Parse ALTO XML file and extract zone (TextBlock) information.
    
    Args:
        file_path: Path to the ALTO XML file
        
    Returns:
        Tuple of (image_path, zones)
        - image_path: Path to the image file
        - zones: List of zone dictionaries with boundaries and labels
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
    
    # Get image path
    source_info = root.find('.//alto:sourceImageInformation', ns)
    image_filename = source_info.find('.//alto:fileName', ns).text if source_info is not None else None
    
    if image_filename:
        # Construct image path relative to XML file
        xml_dir = os.path.dirname(file_path)
        image_path = os.path.join(xml_dir, image_filename)
    else:
        image_path = None
    
    # Load tag labels
    tag_labels = {}
    tags_section = root.find('.//alto:Tags', ns)
    if tags_section is not None:
        for tag in tags_section.findall('.//alto:OtherTag', ns):
            tag_id = tag.get('ID')
            label = tag.get('LABEL')
            if tag_id and label:
                tag_labels[tag_id] = label
    
    # Extract zones (TextBlocks)
    zones = []
    for block in root.findall('.//alto:TextBlock', ns):
        hpos = int(block.get('HPOS', 0))
        vpos = int(block.get('VPOS', 0))
        width = int(block.get('WIDTH', 0))
        height = int(block.get('HEIGHT', 0))
        
        # Get zone label from TAGREFS
        tag_ref = block.get('TAGREFS', 'unknown')
        label = tag_labels.get(tag_ref, 'unknown')
        
        # Try to get polygon from Shape if available
        shape = block.find('.//alto:Shape', ns)
        boundary = None
        if shape is not None:
            polygon = shape.find('.//alto:Polygon', ns)
            if polygon is not None:
                points_str = polygon.get('POINTS', '')
                if points_str:
                    try:
                        if ',' in points_str:
                            # Format: "x1,y1 x2,y2"
                            points = [list(map(int, p.split(','))) for p in points_str.split()]
                        else:
                            # Format: "x1 y1 x2 y2"
                            coords = list(map(int, points_str.split()))
                            points = [[coords[i], coords[i+1]] for i in range(0, len(coords), 2)]
                        
                        if points:
                            boundary = points
                    except Exception as e:
                        print(f"Warning: Could not parse polygon points for block {block.get('ID')}: {e}")
        
        zone_data = {
            'id': block.get('ID'),
            'bbox': [hpos, vpos, hpos + width, vpos + height],
            'label': label,
            'tag_ref': tag_ref
        }
        
        if boundary:
            zone_data['boundary'] = boundary
        
        zones.append(zone_data)
    
    return str(image_path) if image_path else None, zones

