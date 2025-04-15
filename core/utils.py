import os
import glob
import numpy as np
from pathlib import Path
import torch
import wandb
from kraken.lib.xml import XMLPage


def setup_wandb(key=None):
    """
    Configure Weights & Biases for logging.
    
    Args:
        key: Optional WandB API key
    """
    if key:
        wandb.login(key=key)
    else:
        wandb.login()

def get_device():
    """
    Determine the device to use (CUDA or CPU).
    
    Returns:
        String indicating the device ('cuda' or 'cpu')
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def find_files(directory, pattern):
    """
    Find all files matching a pattern in a directory.
    
    Args:
        directory: Directory to search
        pattern: File pattern (e.g., "*.xml", "*.jpg")
        
    Returns:
        List of matching file paths
    """
    return sorted(glob.glob(os.path.join(directory, pattern)))

def ensure_dir(directory):
    """
    Ensure a directory exists, create it if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        Directory path
    """
    os.makedirs(directory, exist_ok=True)
    return directory

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