"""Tests pour les fonctions de manipulation des zones ALTO."""
import pytest
import numpy as np
from src.alto.alto_zones import extract_zones_from_alto, convert_zones_to_boxes


def test_extract_zones_from_alto(sample_alto_path, sample_image_path):
    """Teste l'extraction des zones depuis un fichier ALTO."""
    image_path, zones = extract_zones_from_alto(str(sample_alto_path))
    
    assert zones is not None
    assert len(zones) > 0
    assert "bbox" in zones[0]
    assert "label" in zones[0]


def test_convert_zones_to_boxes(sample_alto_path):
    """Teste la conversion des zones en boxes."""
    _, zones = extract_zones_from_alto(str(sample_alto_path))
    image_size = (640, 480)
    
    boxes = convert_zones_to_boxes(zones, image_size, is_gt=True)
    
    assert isinstance(boxes, np.ndarray)
    assert boxes.shape[1] == 7  # Ground truth format
    
    pred_boxes = convert_zones_to_boxes(zones, image_size, is_gt=False)
    assert pred_boxes.shape[1] == 6  # Prediction format


def test_convert_zones_empty():
    """Teste la conversion avec une liste de zones vide."""
    boxes = convert_zones_to_boxes([], (640, 480), is_gt=True)
    assert boxes.shape == (0, 7)