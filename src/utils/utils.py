import torch
import wandb

import glob
import os
import numpy as np
from sklearn.cluster import DBSCAN

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

def sort_zones_reading_order(zones, eps=300, min_samples=1):
    """
    Sort zones in reading order by detecting columns using clustering.
    
    Algorithm:
    1. Cluster zones by X coordinate (DBSCAN) to detect columns
    2. Sort columns left to right
    3. Within each column, sort zones top to bottom
    
    Args:
        zones: List of zone dictionaries with 'bbox' key [x1, y1, x2, y2]
        eps: Maximum distance (pixels) between zones in the same column
        min_samples: Minimum zones to form a column (default: 1)
        
    Returns:
        Sorted list of zones in reading order
    """
    if not zones:
        return zones
    
    if len(zones) == 1:
        return zones
    
    # Extract center X and top Y coordinates
    data = []
    for zone in zones:
        x1, y1, x2, y2 = zone['bbox']
        center_x = (x1 + x2) / 2
        top_y = y1 
        data.append({'center_x': center_x, 'top_y': top_y})
    
    centers_x = np.array([d['center_x'] for d in data]).reshape(-1, 1)
    
    # Cluster on X axis only to detect columns
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers_x)
    labels = clustering.labels_
    
    # Group zones by column
    columns = {}
    for idx, label in enumerate(labels):
        if label not in columns:
            columns[label] = []
        columns[label].append((idx, zones[idx], data[idx]))
    
    # Sort columns by their average X position (left to right)
    sorted_columns = []
    for label, column_zones in columns.items():
        avg_x = np.mean([d['center_x'] for _, _, d in column_zones])
        sorted_columns.append((avg_x, column_zones))
    
    sorted_columns.sort(key=lambda x: x[0])
    
    # Within each column, sort by top Y coordinate (top to bottom)
    sorted_zones = []
    for _, column_zones in sorted_columns:
        # Sort by top_y (top edge of bbox)
        column_zones.sort(key=lambda x: x[2]['top_y'])
        sorted_zones.extend([zone for _, zone, _ in column_zones])
    
    return sorted_zones

