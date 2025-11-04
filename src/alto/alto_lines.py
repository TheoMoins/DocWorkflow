import os
import glob
import numpy as np
from pathlib import Path
from kraken.lib.xml import XMLPage

from lxml import etree as ET

from src.utils.utils import IGNORED_ZONE_TYPES

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
    parsed = XMLPage(file_path)
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
                'regions': line_regions  # Store region associations
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


def add_lines_to_alto(lines, output_path, alto_path):
    """
    Ajoute des lignes à un fichier ALTO XML existant ou crée un nouveau fichier avec ces lignes.
    
    Args:
        lines: Liste de dictionnaires contenant les informations des lignes (boundary, baseline)
        output_path: Chemin où sauvegarder le fichier ALTO modifié/créé
        alto_path: Chemin vers un fichier ALTO existant        
    Returns:
        True si l'opération a réussi, False sinon
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
            
            x = int(block.get('HPOS', 0))
            y = int(block.get('VPOS', 0))
            w = int(block.get('WIDTH', 0))
            h = int(block.get('HEIGHT', 0))
            block_boxes.append({
                'block': block,
                'bbox': [x, y, x+w, y+h]
            })

            # Remove existing lines ?
            for line in block.findall('.//alto:TextLine', ns):
                block.remove(line)

        # Associer chaque ligne au bloc le plus approprié en utilisant IoU
        for line in lines:
            if 'boundary' in line and line['boundary']:
                # Convertir boundary en bbox pour le calcul IoU
                boundary = np.array(line['boundary'])
                min_x, min_y = boundary.min(axis=0)
                max_x, max_y = boundary.max(axis=0)
                line_bbox = [min_x, min_y, max_x, max_y]
                
                # Trouver le bloc avec le meilleur IoU
                best_iou = 0
                best_block = None
                
                for block_info in block_boxes:
                    iou = calculate_iou(line_bbox, block_info['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_block = block_info['block']
                
                # Si aucun bloc approprié n'est trouvé, utiliser le premier bloc
                if best_block is None and block_boxes:
                    best_block = block_boxes[0]['block']
                
                # Ajouter la ligne au bloc trouvé
                if best_block is not None:
                    _add_line_to_element(best_block, line)
        
        # Sauvegarder le fichier XML modifié
        tree.write(output_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")
        return True
        
    except Exception as e:
        print(f"Error modifying ALTO file: {e}")
        import traceback
        traceback.print_exc()
        return False
