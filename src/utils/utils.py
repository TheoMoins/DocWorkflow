import torch
import wandb

import glob
import os
import numpy as np
from sklearn.cluster import DBSCAN

IGNORED_ZONE_TYPES = {'DigitizationArtefactZone', 'MarginTextZone', 'NumberingZone', 'DropCapitalZone'}

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

def get_zone_first_line_y(zone_info, lines_with_blocks=None):
    """
    Retourne le Y de la première ligne d'une zone, ou le Y du bbox si pas de lignes.
    
    Args:
        zone_info: Dictionnaire d'info de zone avec clé 'bbox'
        lines_with_blocks: Liste des lignes assignées aux zones (optionnel)
        
    Returns:
        Y de la première ligne, ou Y du bbox si pas de lignes ou lines_with_blocks=None
    """
    # Si pas de lignes fournies → fallback sur bbox
    if lines_with_blocks is None:
        return zone_info['bbox'][1]
    
    # Si zone originale avec des lignes
    if zone_info.get('is_original', False):
        zone_lines = [
            item for item in lines_with_blocks 
            if item['block'] == zone_info['block']
        ]
        
        if zone_lines:
            return min(line['y_pos'] for line in zone_lines)
    
    # Fallback : Y du bbox (pseudo-zones ou zones vides)
    return zone_info['bbox'][1]

