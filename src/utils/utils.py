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


def sort_zones_reading_order(zones, lines_with_blocks=None, eps=300, min_samples=1):
    """
    Trier les zones en ordre de lecture (colonnes puis haut-bas).
    
    1. Clustering par X (DBSCAN) pour détecter les colonnes
    2. Trier colonnes de gauche à droite
    3. Dans chaque colonne, trier par Y (première ligne OU bbox selon le mode)
    
    Args:
        zones: Liste de zones (dict avec clé 'bbox': [x1, y1, x2, y2])
        lines_with_blocks: Liste des lignes assignées (optionnel, pour tri avancé)
        eps: Distance max (pixels) entre zones d'une même colonne
        min_samples: Minimum de zones pour former une colonne
        
    Returns:
        Liste triée des zones en ordre de lecture
    """
    if not zones:
        return zones
    
    if len(zones) == 1:
        return zones
    
    # Extraire le centre X de chaque zone
    centers_x = []
    for zone_info in zones:
        bbox = zone_info['bbox']
        center_x = (bbox[0] + bbox[2]) / 2
        centers_x.append(center_x)
    
    centers_x_array = np.array(centers_x).reshape(-1, 1)
    
    # Clustering sur l'axe X pour détecter les colonnes
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers_x_array)
    labels = clustering.labels_
    
    # Grouper les zones par colonne
    columns = {}
    for idx, label in enumerate(labels):
        zone_info = zones[idx]
        
        # Calculer le Y de référence (première ligne OU bbox selon le mode)
        first_line_y = get_zone_first_line_y(zone_info, lines_with_blocks)
        center_x = (zone_info['bbox'][0] + zone_info['bbox'][2]) / 2
        
        if label not in columns:
            columns[label] = []
        
        columns[label].append({
            'zone': zone_info,
            'first_line_y': first_line_y,
            'center_x': center_x
        })
    
    # Trier les colonnes de gauche à droite (par X moyen)
    sorted_columns = []
    for label, zone_list in columns.items():
        avg_x = sum(z['center_x'] for z in zone_list) / len(zone_list)
        sorted_columns.append((avg_x, zone_list))
    
    sorted_columns.sort(key=lambda x: x[0])
    
    # Dans chaque colonne, trier par Y (première ligne), puis X pour départager
    sorted_zones = []
    for _, zone_list in sorted_columns:
        zone_list.sort(key=lambda z: (z['first_line_y'], z['center_x']))
        sorted_zones.extend([z['zone'] for z in zone_list])
    
    return sorted_zones