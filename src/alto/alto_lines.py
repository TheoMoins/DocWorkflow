import os
import glob
import numpy as np
from pathlib import Path

from lxml import etree as ET

from src.utils.utils import IGNORED_ZONE_TYPES
from src.utils.sorting import sort_zones_reading_order
from src.alto import ALTO_NS, ALTO_NS_PREFIX


def _parse_points(points_str: str) -> list:
    """Parse an ALTO POINTS string into [[x, y], ...] pairs.

    Handles both 'x1,y1 x2,y2 ...' (comma-pair) and 'x1 y1 x2 y2 ...' (alternating) formats.
    """
    tokens = points_str.strip().split()
    if not tokens:
        return []
    if ',' in tokens[0]:
        result = []
        for pair in tokens:
            parts = pair.split(',')
            if len(parts) == 2:
                try:
                    result.append([int(float(parts[0])), int(float(parts[1]))])
                except ValueError:
                    continue
        return result
    else:
        result = []
        for i in range(0, len(tokens) - 1, 2):
            try:
                result.append([int(float(tokens[i])), int(float(tokens[i + 1]))])
            except ValueError:
                continue
        return result


def _parse_baseline(baseline_str: str) -> list:
    """Parse an ALTO BASELINE string 'x1 y1 x2 y2 ...' into [[x, y], ...] pairs.

    ALTO BASELINE uses alternating space-separated coordinates (not comma-separated
    pairs like polygon POINTS), e.g. '20 50 620 50' → [[20, 50], [620, 50]].
    """
    parts = baseline_str.strip().split()
    result = []
    for i in range(0, len(parts) - 1, 2):
        try:
            result.append([int(float(parts[i])), int(float(parts[i + 1]))])
        except ValueError:
            continue
    return result

# Non-standard ALTO attribute used to carry a detector confidence on a TextLine.
# ALTO v4 defines no confidence attribute at line level (only String/@WC at word
# level), so predictions store theirs here and the scorer reads it back.
CONFIDENCE_ATTR = 'CONFIDENCE'


def _parse_confidence(line_elem, default=1.0):
    """Read CONFIDENCE_ATTR from a TextLine element, falling back to `default`."""
    raw = line_elem.get(CONFIDENCE_ATTR)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


# Default vertical position of the baseline inside a line polygon, as a fraction of
# the local polygon height (0 = top edge, 1 = bottom edge).
# Calibrated on the CATMuS-medieval annotations (33k lines: median 0.670, mean 0.664,
# std 0.098) and cross-checked on the CATMuS+LostMa segmentation test set (median 0.653).
DEFAULT_BASELINE_RATIO = 0.67


def polygon_to_baseline(polygon, ratio=DEFAULT_BASELINE_RATIO, n_points=8,
                        straight_tol=1.0):
    """
    Derive a baseline polyline from a line polygon.

    The baseline is placed at a fixed fraction of the *local* polygon height, sampled
    at several x positions, so it follows the slant and the curvature of the line
    instead of being a single horizontal segment through the bounding box.

    Args:
        polygon: [[x, y], ...] polygon of the line (at least 3 points)
        ratio: vertical position inside the local polygon height (see
            DEFAULT_BASELINE_RATIO)
        n_points: number of samples along the x axis
        straight_tol: if every intermediate point lies within this many pixels of the
            straight segment joining the extremities, the baseline is collapsed to
            2 points (most annotated baselines are straight)

    Returns:
        [[x, y], ...] baseline, or None if the polygon is degenerate
    """
    pts = np.asarray(polygon, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return None

    x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
    if x_max - x_min < 1.0:
        return None

    closed = np.vstack([pts, pts[:1]])
    xa, ya = closed[:-1, 0], closed[:-1, 1]
    xb, yb = closed[1:, 0], closed[1:, 1]

    xs = np.linspace(x_min, x_max, max(int(n_points), 2))
    tops = np.full(xs.shape, np.nan)
    bots = np.full(xs.shape, np.nan)

    eps = 1e-9
    lo = np.minimum(xa, xb)
    hi = np.maximum(xa, xb)
    dx = xb - xa
    vertical = np.abs(dx) < eps

    for i, x in enumerate(xs):
        hit = (lo - eps <= x) & (x <= hi + eps)
        if not hit.any():
            continue
        ys = []
        # Slanted edges: interpolate the crossing.
        slanted = hit & ~vertical
        if slanted.any():
            t = (x - xa[slanted]) / dx[slanted]
            ys.append(ya[slanted] + t * (yb[slanted] - ya[slanted]))
        # Vertical edges: both endpoints are on the sweep line.
        vert = hit & vertical
        if vert.any():
            ys.append(ya[vert])
            ys.append(yb[vert])
        ys = np.concatenate(ys)
        tops[i] = ys.min()
        bots[i] = ys.max()

    valid = ~np.isnan(tops)
    if valid.sum() < 2:
        return None
    if not valid.all():
        tops = np.interp(xs, xs[valid], tops[valid])
        bots = np.interp(xs, xs[valid], bots[valid])

    ys = tops + ratio * (bots - tops)

    # Collapse to a 2-point baseline when the polyline is straight enough.
    if len(xs) > 2:
        straight = np.interp(xs, [xs[0], xs[-1]], [ys[0], ys[-1]])
        if np.abs(ys - straight).max() <= straight_tol:
            xs, ys = xs[[0, -1]], ys[[0, -1]]

    return [[int(round(x)), int(round(y))] for x, y in zip(xs, ys)]


def bbox_to_baseline(x1, y1, x2, y2, ratio=DEFAULT_BASELINE_RATIO):
    """
    Fallback baseline for a line known only by its bounding box (pure detection).

    Horizontal by construction — a box carries no slant information — but placed at
    the calibrated height rather than at mid-height.
    """
    y = int(round(y1 + ratio * (y2 - y1)))
    return [[int(x1), y], [int(x2), y]]


def normalize_box(box, width, height, scale=100):
    """
    Normalize box coordinates onto an integer grid.

    Warning: this quantisation is only usable for layout *zones*, which cover a large
    fraction of the page. It must not be used for text *lines*: a line is ~2 units tall
    on the default 0-100 grid, so the truncation dominates the IoU. Line scoring works
    in pixel coordinates (see convert_lines_to_boxes).
    
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

def calculate_containment(box, container):
    """
    Fraction of `box` that lies inside `container` (0 to 1).

    Unlike IoU, this does not shrink when the container is large, which is what is
    needed to decide which layout zone a thin text line belongs to.

    Args:
        box: [x1, y1, x2, y2]
        container: [x1, y1, x2, y2]

    Returns:
        Containment ratio between 0 and 1
    """
    x1 = max(box[0], container[0])
    y1 = max(box[1], container[1])
    x2 = min(box[2], container[2])
    y2 = min(box[3], container[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])

    return intersection / box_area if box_area > 0 else 0.0


def read_lines_geometry(file_path):
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
        tree = ET.parse(file_path)
        root = tree.getroot()
    except Exception:
        return ('', [], {})

    ns = ALTO_NS_PREFIX

    # Image filename
    file_elem = root.find('.//alto:sourceImageInformation/alto:fileName', ns)
    image_filename = file_elem.text.strip() if file_elem is not None and file_elem.text else ''
    alto_dir = os.path.dirname(os.path.abspath(file_path))
    image_path = os.path.join(alto_dir, image_filename) if image_filename else ''

    # Build tag ID → label mapping
    tag_labels = {}
    for tag in root.findall('.//alto:Tags/*', ns):
        tag_id = tag.get('ID')
        label = tag.get('LABEL')
        if tag_id and label:
            tag_labels[tag_id] = label

    def resolve_tagrefs(tagrefs_str):
        return [tag_labels.get(ref, ref) for ref in tagrefs_str.split() if ref]

    # Extract regions (TextBlocks)
    regions = {}
    block_region_type = {}

    for block in root.findall('.//alto:TextBlock', ns):
        block_id = block.get('ID', '')
        labels = resolve_tagrefs(block.get('TAGREFS', ''))
        region_type = labels[0] if labels else ''

        if region_type in IGNORED_ZONE_TYPES:
            continue

        block_region_type[block_id] = region_type

        polygon = block.find('alto:Shape/alto:Polygon', ns)
        if polygon is not None:
            boundary = _parse_points(polygon.get('POINTS', ''))
        else:
            hpos = int(float(block.get('HPOS', 0)))
            vpos = int(float(block.get('VPOS', 0)))
            w = int(float(block.get('WIDTH', 0)))
            h = int(float(block.get('HEIGHT', 0)))
            boundary = [[hpos, vpos], [hpos+w, vpos], [hpos+w, vpos+h], [hpos, vpos+h]] if w and h else []

        if boundary:
            regions.setdefault(region_type, []).append(boundary)

    # Extract lines from non-ignored TextBlocks
    lines = []
    for block in root.findall('.//alto:TextBlock', ns):
        block_id = block.get('ID', '')
        if block_id not in block_region_type:
            continue

        region = block_region_type[block_id]

        for line_elem in block.findall('alto:TextLine', ns):
            baseline = _parse_baseline(line_elem.get('BASELINE', ''))

            polygon = line_elem.find('alto:Shape/alto:Polygon', ns)
            boundary = _parse_points(polygon.get('POINTS', '')) if polygon is not None else None

            # A line is usable as soon as it carries one of the two geometries.
            # Requiring a BASELINE here used to silently drop polygon-only lines,
            # which made the ground truth read differently depending on the metric.
            if len(baseline) < 2 and (not boundary or len(boundary) < 3):
                continue

            text = ' '.join(
                s.get('CONTENT', '')
                for s in line_elem.findall('.//alto:String', ns)
                if s.get('CONTENT')
            )

            tags = set(resolve_tagrefs(line_elem.get('TAGREFS', '')))

            lines.append({
                'id': line_elem.get('ID', ''),
                'baseline': baseline,
                'boundary': boundary,
                'tags': tags,
                'regions': [region],
                'text': text,
                'confidence': _parse_confidence(line_elem),
            })

    return str(image_path), lines, regions


def read_lines(file_path: str, with_text: bool = False) -> list:
    """
    Single line reader used for *every* scoring metric, on both the ground truth and
    the predictions. Having one parser is what guarantees that mAP and ZoneMapAlt
    score the same set of lines and that GT and predictions are read symmetrically.

    Geometry fallback chain:
      Shape/Polygon  → boundary polygon
      BASELINE       → baseline polyline (generates bbox)
      HPOS/VPOS/WIDTH/HEIGHT → synthetic 2-point baseline

    Skips TextBlocks whose region type is in IGNORED_ZONE_TYPES, so that lines the
    protocol excludes are dropped identically on both sides.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except Exception:
        return []

    ns = ALTO_NS_PREFIX

    tag_labels = {}
    for tag in root.findall('.//alto:Tags/*', ns):
        tag_id = tag.get('ID')
        label = tag.get('LABEL')
        if tag_id and label:
            tag_labels[tag_id] = label

    ignored_block_ids = set()
    for block in root.findall('.//alto:TextBlock', ns):
        refs = block.get('TAGREFS', '')
        labels = [tag_labels.get(r, r) for r in refs.split() if r]
        if labels and labels[0] in IGNORED_ZONE_TYPES:
            ignored_block_ids.add(block.get('ID', ''))

    lines = []

    for block in root.findall('.//alto:TextBlock', ns):
        if block.get('ID', '') in ignored_block_ids:
            continue
        for line_elem in block.findall('alto:TextLine', ns):
            polygon = line_elem.find('alto:Shape/alto:Polygon', ns)
            boundary = _parse_points(polygon.get('POINTS', '')) if polygon is not None else None

            baseline_str = line_elem.get('BASELINE', '')
            baseline = _parse_baseline(baseline_str) if baseline_str else []

            # A boundary is only usable as a polygon from 3 points on; below that
            # it carries no area and downstream geometry (ZoneMapAlt) breaks on it.
            if boundary is not None and len(boundary) < 3:
                boundary = None

            if not boundary and len(baseline) < 2:
                try:
                    hpos = int(float(line_elem.get('HPOS', 0)))
                    vpos = int(float(line_elem.get('VPOS', 0)))
                    w    = int(float(line_elem.get('WIDTH', 0)))
                    h    = int(float(line_elem.get('HEIGHT', 0)))
                except (ValueError, TypeError):
                    continue
                if w <= 0 or h <= 0:
                    continue
                mid_y = vpos + h // 2
                baseline = [[hpos, mid_y], [hpos + w, mid_y]]

            if not boundary and len(baseline) < 2:
                continue

            text = ''
            if with_text:
                text = ' '.join(
                    s.get('CONTENT', '')
                    for s in line_elem.findall('.//alto:String', ns)
                    if s.get('CONTENT')
                )

            lines.append({
                'id':         line_elem.get('ID', ''),
                'baseline':   baseline,
                'boundary':   boundary,
                'text':       text,
                'confidence': _parse_confidence(line_elem),
            })

    return lines


# Backwards-compatible alias: this reader is no longer specific to ZoneMap.
read_lines_for_zonemap = read_lines


def convert_lines_to_boxes(lines, image_size=None, is_gt=True, scale=None, buffer=5):
    """
    Convert lines (baselines with boundaries) to bounding box format.

    Coordinates stay in **pixels** by default. The previous 0-100 integer grid was
    inherited from YALTAi, where it fits layout zones (10-80 % of the page); a text
    line is ~2 units tall on that grid, so truncation, not prediction quality, drove
    the IoU. mAP is scale-invariant, so pixel coordinates lose nothing.

    Args:
        lines: List of line dictionaries with baselines and boundaries
        image_size: (width, height) of the image — only needed when `scale` is set
        is_gt: Whether these are ground truth boxes (True) or predictions (False)
        scale: Legacy quantisation grid (e.g. 100). None (default) = pixel coordinates.
        buffer: Half-height, in pixels, of the box synthesised around a line that has
            no polygon

    Returns:
        Numpy array of boxes
    """
    if scale is not None:
        if image_size is None:
            raise ValueError("image_size is required when scale is set")
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
            min_y -= buffer
            max_y += buffer
            boundary = np.array([[min_x, min_y], [max_x, min_y], 
                                [max_x, max_y], [min_x, max_y]])
        
        # Get bounding box coordinates
        min_x, min_y = boundary.min(axis=0)
        max_x, max_y = boundary.max(axis=0)
        
        if scale is None:
            x1, y1, x2, y2 = float(min_x), float(min_y), float(max_x), float(max_y)
        else:
            # Legacy quantised grid — kept only to reproduce historical numbers.
            x1 = int(scale * min_x / width)
            y1 = int(scale * min_y / height)
            x2 = int(scale * max_x / width)
            y2 = int(scale * max_y / height)
        
        # Use 0 as class_id for all lines (we don't differentiate line types for now)
        class_id = 0
        
        if is_gt:
            # For ground truth: [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
            boxes.append([x1, y1, x2, y2, class_id, 0, 0])
        else:
            # For predictions: [xmin, ymin, xmax, ymax, class_id, confidence_score].
            # The detector score drives the precision/recall curve the AP integrates;
            # a constant here collapses that curve to a single arbitrary ordering.
            confidence = float(line.get('confidence', 1.0))
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
    line_element = ET.SubElement(parent_element, f"{{{ALTO_NS}}}TextLine")
    
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
        shape = ET.SubElement(line_element, f"{{{ALTO_NS}}}Shape")
        polygon = ET.SubElement(shape, f"{{{ALTO_NS}}}Polygon")
        points = " ".join([f"{int(p[0])},{int(p[1])}" for p in boundary])
        polygon.set('POINTS', points)
    
    # Ajouter la baseline si disponible
    if 'baseline' in line and line['baseline']:
        baseline = line['baseline']
        baseline_str = " ".join(f"{int(p[0])} {int(p[1])}" for p in baseline)
        line_element.set('BASELINE', baseline_str)
    
    # Conserver le score de confiance du détecteur (cf. CONFIDENCE_ATTR)
    if line.get('confidence') is not None:
        line_element.set(CONFIDENCE_ATTR, f"{float(line['confidence']):.6f}")
    
    return line_element


def add_lines_to_alto(lines, output_path, alto_path, reading_order="dbscan",
                      orphan_policy="pseudo_block"):
    """
    Ajoute des lignes à un fichier ALTO XML existant.

    Les lignes sont affectées au TextBlock de meilleur IoU, **y compris** les zones
    de IGNORED_ZONE_TYPES : une ligne prédite dans une MarginTextZone hérite ainsi du
    TAGREFS de cette zone et sera écartée au scoring exactement comme la ligne de
    référence correspondante. Auparavant ces zones étaient ignorées à l'affectation
    *et* supprimées du fichier de sortie, si bien que ces prédictions survivaient sans
    TAGREFS et étaient comptées en faux positifs alors que la référence, elle, était
    filtrée.

    Args:
        lines: lignes prédites (dicts avec 'boundary', 'baseline', éventuellement
            'confidence')
        output_path: fichier ALTO à écrire
        alto_path: fichier ALTO source (layout)
        reading_order: méthode d'ordonnancement des zones
        orphan_policy: que faire des lignes ne tombant dans aucune zone —
            "pseudo_block" (défaut) crée une zone mono-ligne ; "drop" les supprime,
            ce qui contraint le modèle au layout comme le fait Kraken par
            construction (protocole symétrique pour le benchmark).
    """
    if orphan_policy not in ("pseudo_block", "drop"):
        raise ValueError(f"Unknown orphan_policy: {orphan_policy!r}")
    try:
        # Extraire les informations du fichier ALTO existant
        image_file, _, regions = read_lines_geometry(alto_path)
        
        # Parser le fichier XML
        # modified to fix spacing
        parser = ET.XMLParser(remove_blank_text=True)
        tree = ET.parse(alto_path, parser)
        root = tree.getroot()
        ns = ALTO_NS_PREFIX
        
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
        
        # Préparer les blocs avec leurs boîtes délimitantes.
        # Les IGNORED_ZONE_TYPES sont conservées comme cibles d'affectation : c'est ce
        # qui rend le filtrage au scoring symétrique entre référence et prédiction.
        block_boxes = []
        for block in text_blocks:
            tag_ref = block.get('TAGREFS', '')
            block_label = tag_labels.get(tag_ref, '')
            
            x = int(float(block.get('HPOS', 0)))
            y = int(float(block.get('VPOS', 0)))
            w = int(float(block.get('WIDTH', 0)))
            h = int(float(block.get('HEIGHT', 0)))
            block_boxes.append({
                'block': block,
                'bbox': [x, y, x+w, y+h],
                'is_original': True,
                'label': block_label
            })

            # Remove existing lines
            for line in block.findall('.//alto:TextLine', ns):
                block.remove(line)
        
        # Une ligne appartient à la zone qui en contient la plus grande part.
        # L'IoU ne convient pas ici : une ligne fine a un IoU d'autant plus élevé que
        # la zone est petite, si bien qu'une ligne de MainZone effleurant une petite
        # zone marginale était attribuée à cette dernière. Le taux de contenance
        # (aire de la ligne dans la zone / aire de la ligne) est la bonne mesure.
        CONTAINMENT_THRESHOLD = 0.001
        
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
            best_score = 0
            best_block = None
            
            for block_info in block_boxes:
                if not block_info['is_original']:  # Ignorer les pseudo-zones déjà créées
                    continue
                score = calculate_containment(line_bbox, block_info['bbox'])
                if score > best_score:
                    best_score = score
                    best_block = block_info
            
            # Si la zone contient une part suffisante de la ligne, l'y assigner
            if best_score >= CONTAINMENT_THRESHOLD and best_block is not None:
                lines_with_blocks.append({
                    'line': line,
                    'block': best_block['block'],
                    'block_bbox': best_block['bbox'],
                    'y_pos': line_y
                })
            elif orphan_policy == "pseudo_block":
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
