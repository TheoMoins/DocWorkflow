"""
ALTO XML Generator from YOLO predictions.
Based on yolalto (https://github.com/ponteineptique/yolalto) by Thibault Clérice.

Adapted for the Document Analysis Framework.
"""

from lxml import etree
import itertools
import os
import hashlib



def create_tag_id(label, prefix="BT"):
    """
    Create a stable hash-based tag ID with a given prefix.
    
    Args:
        label: The label text
        prefix: Prefix for the ID (default: "BT" for Block Type)
        
    Returns:
        A stable ID for the label
    """
    h = hashlib.sha1(label.encode()).hexdigest()[:6]
    numeric_part = int(h, 16) % 100000
    return f"{prefix}{numeric_part}"


def parse_yolo_results(results):
    """
    Parse YOLO results to detections and image dimensions.
    
    Args:
        results: YOLOv8 model predictions
        
    Returns:
        List of tuples, each containing detections and image dimensions
    """
    out = []
    for result in results:
        detections = []
        h, w = result.orig_shape
        for box in result.boxes:
            label = result.names[int(box.cls)]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({'label': label, 'bbox': [x1, y1, x2, y2]})
        out.append((detections, (w, h)))
    return out


def bbox_to_polygon(bbox):
    """
    Convert a bbox to a 4-point polygon for ALTO.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        List of polygon points [x1, y1, x2, y1, x2, y2, x1, y2]
    """
    x1, y1, x2, y2 = map(int, bbox)
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def bbox_baseline(bbox):
    """
    Generate a baseline from left to right, in the middle of the bbox.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        Baseline points [x1, center_y, x2, center_y]
    """
    x1, y1, x2, y2 = map(int, bbox)
    center_y = int((y1 + y2) // 2)
    return [x1, center_y, x2, center_y]


def remove_duplicates(detections):
    """
    Remove duplicate detections based on IoU threshold.
    
    Args:
        detections: List of detection dictionaries with 'bbox' key
        
    Returns:
        Filtered list of detections
    """
    from core.utils import calculate_iou  # Utiliser la fonction existante
    
    bboxes = {idx: det['bbox'] for idx, det in enumerate(detections)}
    to_delete = set()
    checked_pairs = set()
    
    for (id1, box1), (id2, box2) in itertools.combinations(bboxes.items(), 2):
        pair_key = tuple(sorted((id1, id2)))
        if pair_key in checked_pairs:
            continue
        checked_pairs.add(pair_key)
        score = calculate_iou(box1, box2)
        if score > 0.5:
            # Delete the detection with the smaller box
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            delete_id = id1 if area1 < area2 else id2
            to_delete.add(delete_id)
    
    return [det for idx, det in enumerate(detections) if idx not in to_delete]


def assign_zones(detections):
    """
    Extract zones from detections.
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        List of zones with additional index information
    """
    zones = [d for d in detections if d["label"].endswith("Zone")]
    return [{"idx": idx, **zone} for (idx, zone) in enumerate(zones)]


def create_alto_xml(detections, image_path, dimensions):
    """
    Generate ALTO XML from detections.
    
    Args:
        detections: List of detection dictionaries
        image_path: Path to the original image
        dimensions: Image dimensions (width, height)
        
    Returns:
        lxml Element containing the ALTO XML
    """
    # Build tag registry for all unique labels
    tag_registry = {}
    tags = set()
    
    for det in detections:
        label = det["label"]
        if label not in tag_registry:
            tag_id = create_tag_id(label, "BT")
            tag_registry[label] = tag_id
            tags.add((tag_id, label, "block type"))
    
    NSMAP = {
        None: "http://www.loc.gov/standards/alto/ns-v4#",
        "xsi": "http://www.w3.org/2001/XMLSchema-instance"
    }
    
    alto = etree.Element(
        "alto",
        nsmap=NSMAP,
        attrib={
            "{http://www.w3.org/2001/XMLSchema-instance}schemaLocation":
                "http://www.loc.gov/standards/alto/ns-v4# http://www.loc.gov/standards/alto/v4/alto-4-2.xsd"
        }
    )
    
    # Description section
    description = etree.SubElement(alto, "Description")
    etree.SubElement(description, "MeasurementUnit").text = "pixel"
    
    source_info = etree.SubElement(description, "sourceImageInformation")
    filename = os.path.basename(image_path)
    etree.SubElement(source_info, "fileName").text = filename
    etree.SubElement(source_info, "fileIdentifier").text = filename
    
    # Processing section
    processing = etree.SubElement(description, "OCRProcessing")
    processing_step = etree.SubElement(processing, "ocrProcessingStep")
    software = etree.SubElement(processing_step, "processingSoftware")
    etree.SubElement(software, "softwareName").text = "Document Analysis Framework"
    
    # Tags section
    tags_elem = etree.SubElement(alto, "Tags")
    for tag_id, label, tag_type in sorted(tags):
        etree.SubElement(tags_elem, "OtherTag", ID=tag_id, LABEL=label, DESCRIPTION=f"{tag_type} {label}")
    
    # Layout section
    width, height = dimensions
    layout = etree.SubElement(alto, "Layout")
    page = etree.SubElement(layout, "Page", ID="page1", PHYSICAL_IMG_NR="1", HEIGHT=str(height), WIDTH=str(width))
    print_space = etree.SubElement(page, "PrintSpace", HEIGHT=str(height), WIDTH=str(width), VPOS="0", HPOS="0")
    
    # Add zones
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = map(int, det['bbox'])
        tb = etree.SubElement(
            print_space, 
            "TextBlock", 
            ID=f"block_{i}", 
            HPOS=str(x1), 
            VPOS=str(y1),
            WIDTH=str(x2 - x1), 
            HEIGHT=str(y2 - y1), 
            TAGREFS=tag_registry[det["label"]]
        )
        shape = etree.SubElement(tb, "Shape")
        polygon_points = " ".join([str(p) for p in bbox_to_polygon(det["bbox"])])
        etree.SubElement(shape, "Polygon", POINTS=polygon_points)
    
    return alto


def save_alto_xml(alto_element, output_path):
    """
    Write ALTO XML to file.
    
    Args:
        alto_element: lxml Element containing the ALTO XML
        output_path: Path to save the XML file
    """
    tree = etree.ElementTree(alto_element)
    tree.write(output_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")