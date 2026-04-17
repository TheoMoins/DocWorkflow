import os
import glob
import numpy as np
from pathlib import Path
from kraken.lib.xml import XMLPage

from lxml import etree as ET

from src.utils.utils import IGNORED_ZONE_TYPES
from src.utils.sorting import sort_zones_reading_order


def normalize_box(box, width, height, scale=100):
    """
    Normalize box coordinates.
    
    Args:
        box: Numpy array of coordinates [x1, y1, x2, y2]
        width: Image width
        height: Image height
        scale: Scale factor (default 100 for percentage)
        
    Returns:
        Numpy array of normalized coordinates
    """
    x1, y1, x2, y2 = box
    return np.array([
        int(scale * x1 / width),
        int(scale * y1 / height),
        int(scale * x2 / width),
        int(scale * y2 / height)
    ])

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two boxes.
    
    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]
        
    Returns:
        IoU score between 0 and 1
    """
    # Intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate box areas
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union
    union = box1_area + box2_area - intersection
    
    # Calculate IoU
    return intersection / union if union > 0 else 0.0

def extract_lines_from_alto(file_path):
    """
    Parse ALTO XML file and extract line and region information.
    
    Args:
        file_path: Path to the ALTO XML file
        
    Returns:
        Tuple of (image_path, lines, regions)
        - image_path: Path to the image file
        - lines: List of line dictionaries with boundaries, baselines and region info
        - regions: Dictionary of regions by type
    """
    try:
    	parsed = XMLPage(file_path)
    except:
        return ('',[],{})
    image_path = parsed.imagename
    
    # Extract regions from the ALTO file
    regions = {}
    region_polygons = {}  # Store region polygons by ID for spatial containment check
    
    for region_type, region_list in parsed.regions.items():
        if region_type in IGNORED_ZONE_TYPES:
            continue

        if region_type not in regions:
            regions[region_type] = []
        
        for region_obj in region_list:
            if region_obj.boundary:
                regions[region_type].append(region_obj.boundary)
                region_polygons[region_obj.id] = region_obj.boundary
    
    # Extract lines from the ALTO file and associate with regions
    lines = []
    for line_id, line_obj in parsed.lines.items():
        if line_obj.baseline and len(line_obj.baseline) >= 2:
            # Check which regions this line belongs to
            line_regions = []
            if hasattr(line_obj, 'regions') and line_obj.regions:
                line_regions = line_obj.regions

            if not line_regions or any(region in IGNORED_ZONE_TYPES for region in line_regions):
                continue
            
            lines.append({
                'id': line_id,
                'baseline': line_obj.baseline,
                'boundary': line_obj.boundary if line_obj.boundary else None,
                'tags': line_obj.tags,
                'regions': line_regions,  # Store region associations
                'text': line_obj.text
            })
    
    return str(image_path), lines, regions


def convert_lines_to_boxes(lines, image_size, is_gt=True):
    """
    Convert lines (baselines with boundaries) to bounding box format
    
    Args:
        lines: List of line dictionaries with baselines and boundaries
        image_size: (width, height) of the image
        is_gt: Whether these are ground truth boxes (True) or predictions (False)
        
    Returns:
        Numpy array of boxes
    """
    width, height = image_size
    boxes = []
    
    for i, line in enumerate(lines):
        # Use line boundary if available, otherwise create boundary from baseline
        if line.get('boundary'):
            boundary = np.array(line['boundary'])
        else:
            # Create a simple boundary around the baseline
            baseline = np.array(line['baseline'])
            min_x, min_y = baseline.min(axis=0)
            max_x, max_y = baseline.max(axis=0)
            # Add a small buffer around the baseline
            buffer = 5
            min_y -= buffer
            max_y += buffer
            boundary = np.array([[min_x, min_y], [max_x, min_y], 
                                [max_x, max_y], [min_x, max_y]])
        
        # Get bounding box coordinates
        min_x, min_y = boundary.min(axis=0)
        max_x, max_y = boundary.max(axis=0)
        
        # Normalize coordinates to 0-100 scale for consistency with YALTAi metrics
        x1 = int(100 * min_x / width)
        y1 = int(100 * min_y / height)
        x2 = int(100 * max_x / width)
        y2 = int(100 * max_y / height)
        
        # Use 0 as class_id for all lines (we don't differentiate line types for now)
        class_id = 0
        
        if is_gt:
            # For ground truth: [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
            boxes.append([x1, y1, x2, y2, class_id, 0, 0])
        else:
            # For predictions: [xmin, ymin, xmax, ymax, class_id, confidence_score]
            confidence = 1.0  # Use 1.0 as default confidence score
            boxes.append([x1, y1, x2, y2, class_id, confidence])
    
    return np.array(boxes) if boxes else np.zeros((0, 7 if is_gt else 6))


def _add_line_to_element(parent_element, line, line_id=None, tag_id="LT1"):
    """
    Ajoute une ligne TextLine à un élément parent dans un document ALTO XML.
    
    Args:
        parent_element: Élément XML parent (TextBlock)
        line: Dictionnaire contenant les informations de la ligne (boundary, baseline)
        line_id: ID à attribuer à la ligne (généré automatiquement si None)
        tag_id: ID de la balise à référencer (TAGREFS)
        
    Returns:
        L'élément TextLine créé
    """
    # Créer l'élément TextLine
    ns = 'http://www.loc.gov/standards/alto/ns-v4#'
    line_element = ET.SubElement(parent_element, f"{{{ns}}}TextLine")
    
    # Ajouter l'ID
    if line_id is None and 'id' in line:
        line_id = line['id']
    elif line_id is None:
        line_id = f"line_{id(line)}"
    
    line_element.set('ID', line_id)
    line_element.set('TAGREFS', tag_id)
    
    # Ajouter les informations de boundary si disponibles
    if 'boundary' in line and line['boundary']:
        boundary = np.array(line['boundary'])
        min_x, min_y = boundary.min(axis=0)
        max_x, max_y = boundary.max(axis=0)
        
        line_element.set('HPOS', str(int(min_x)))
        line_element.set('VPOS', str(int(min_y)))
        line_element.set('WIDTH', str(int(max_x - min_x)))
        line_element.set('HEIGHT', str(int(max_y - min_y)))
        
        # Ajouter Shape avec Polygon
        shape = ET.SubElement(line_element, f"{{{ns}}}Shape")
        polygon = ET.SubElement(shape, f"{{{ns}}}Polygon")
        points = " ".join([f"{int(p[0])},{int(p[1])}" for p in boundary])
        polygon.set('POINTS', points)
    
    # Ajouter la baseline si disponible
    if 'baseline' in line and line['baseline']:
        baseline = line['baseline']
        baseline_str = " ".join([f"{int(p[0])},{int(p[1])}" for p in baseline])
        line_element.set('BASELINE', baseline_str)
    
    return line_element


def add_lines_to_alto(lines, output_path, alto_path, reading_order="dbscan"):
    """
    Ajoute des lignes à un fichier ALTO XML existant.
    Les lignes sans zone correspondante (IoU faible) deviennent des zones mono-ligne.
    """
    try:
        # Extraire les informations du fichier ALTO existant
        image_file, _, regions = extract_lines_from_alto(alto_path)
        
        # Parser le fichier XML
        tree = ET.parse(alto_path)
        root = tree.getroot()
        ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
        
        # Vérifier qu'il y a au moins un bloc de texte
        text_blocks = root.findall('.//alto:TextBlock', ns)
        if not text_blocks:
            print(f"Warning: No TextBlocks found in {alto_path}")

        # Récupérer les labels des tags
        tag_labels = {}
        tags_section = root.find('.//alto:Tags', ns)
        if tags_section is not None:
            for tag in tags_section.findall('.//alto:OtherTag', ns):
                tag_id = tag.get('ID')
                label = tag.get('LABEL')
                if tag_id and label:
                    tag_labels[tag_id] = label
        
        # Préparer les blocs avec leurs boîtes délimitantes
        block_boxes = []
        for block in text_blocks:
            tag_ref = block.get('TAGREFS', '')
            block_label = tag_labels.get(tag_ref, '')
            
            # Ignorer les DropCapitalZone et autres zones non-textuelles
            if block_label in IGNORED_ZONE_TYPES:
                continue
            
            x = int(float(block.get('HPOS', 0)))
            y = int(float(block.get('VPOS', 0)))
            w = int(float(block.get('WIDTH', 0)))
            h = int(float(block.get('HEIGHT', 0)))
            block_boxes.append({
                'block': block,
                'bbox': [x, y, x+w, y+h],
                'is_original': True
            })

            # Remove existing lines
            for line in block.findall('.//alto:TextLine', ns):
                block.remove(line)
        
        IOU_THRESHOLD = 0.001  # Seuil minimum d'IoU pour considérer qu'une ligne appartient à une zone
        
        lines_with_blocks = []  # Liste de (line, block, y_pos)
        orphan_lines = []       # Lignes sans zone correspondante
        
        for line in lines:
            if 'boundary' not in line or not line['boundary']:
                continue
                
            # Calculer bbox de la ligne
            boundary = np.array(line['boundary'])
            min_x, min_y = boundary.min(axis=0)
            max_x, max_y = boundary.max(axis=0)
            line_bbox = [min_x, min_y, max_x, max_y]
            line_y = (min_y + max_y) / 2
            
            # Chercher le meilleur bloc
            best_iou = 0
            best_block = None
            
            for block_info in block_boxes:
                if not block_info['is_original']:  # Ignorer les pseudo-zones déjà créées
                    continue
                iou = calculate_iou(line_bbox, block_info['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_block = block_info
            
            # Si IoU suffisant, assigner à la zone
            if best_iou >= IOU_THRESHOLD and best_block is not None:
                lines_with_blocks.append({
                    'line': line,
                    'block': best_block['block'],
                    'block_bbox': best_block['bbox'],
                    'y_pos': line_y
                })
            else:
                # Ligne orpheline : créer une pseudo-zone mono-ligne
                orphan_lines.append({
                    'line': line,   
                    'bbox': line_bbox,
                    'y_pos': line_y
                })
        
        # === Créer des pseudo-zones pour les lignes orphelines ===
        
        print_space = root.find('.//alto:PrintSpace', ns)
        if print_space is None:
            print(f"Warning: No PrintSpace found in {alto_path}")
            return False
        
        for idx, orphan in enumerate(orphan_lines):
            block_boxes.append({
                'block': None,
                'bbox': orphan['bbox'],
                'is_original': False,
                'orphan_data': orphan
            })
        
        sorted_blocks = sort_zones_reading_order(
            block_boxes, 
            lines_with_blocks, 
            eps=200,
            method=reading_order
        )
                
        # Supprimer tous les TextBlocks existants
        for text_block in list(print_space.findall(f"{{{ns['alto']}}}TextBlock")):
            print_space.remove(text_block)
        
        # Recréer dans l'ordre trié
        pseudo_counter = 0
        for block_info in sorted_blocks:
            if block_info['is_original']:
                # Zone originale
                print_space.append(block_info['block'])
                
                # Ajouter lignes triées par Y
                block_lines = [
                    item for item in lines_with_blocks 
                    if item['block'] == block_info['block']
                ]
                block_lines.sort(key=lambda x: x['y_pos'])
                
                for line_data in block_lines:
                    _add_line_to_element(block_info['block'], line_data['line'])
            else:
                # Pseudo-zone
                bbox = block_info['bbox']
                pseudo_block = ET.SubElement(print_space, f"{{{ns['alto']}}}TextBlock")
                pseudo_block.set('ID', f"pseudo_block_{pseudo_counter}")
                pseudo_block.set('HPOS', str(int(bbox[0])))
                pseudo_block.set('VPOS', str(int(bbox[1])))
                pseudo_block.set('WIDTH', str(int(bbox[2] - bbox[0])))
                pseudo_block.set('HEIGHT', str(int(bbox[3] - bbox[1])))
                
                orphan_data = block_info['orphan_data']
                _add_line_to_element(pseudo_block, orphan_data['line'])
                
                pseudo_counter += 1
        
        tree.write(output_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")
        return True
        
    except Exception as e:
        print(f"Error modifying ALTO file: {e}")
        import traceback
        traceback.print_exc()
        return False
