"""Tests pour les fonctions de manipulation des lignes ALTO."""
import pytest
import numpy as np
from lxml import etree as ET
from src.alto.alto_lines import (
    extract_lines_from_alto,
    convert_lines_to_boxes,
    calculate_iou,
    normalize_box,
    add_lines_to_alto,
    _add_line_to_element
)


@pytest.fixture
def sample_alto_with_lines(temp_dir, sample_image_path):
    """Crée un fichier ALTO avec des lignes de texte."""
    ns = "http://www.loc.gov/standards/alto/ns-v4#"
    alto = ET.Element(
        "alto",
        nsmap={None: ns, "xsi": "http://www.w3.org/2001/XMLSchema-instance"}
    )
    
    # Description
    desc = ET.SubElement(alto, "Description")
    ET.SubElement(desc, "MeasurementUnit").text = "pixel"
    source = ET.SubElement(desc, "sourceImageInformation")
    ET.SubElement(source, "fileName").text = sample_image_path.name
    
    # Tags
    tags = ET.SubElement(alto, "Tags")
    ET.SubElement(tags, "OtherTag", ID="BT1", LABEL="MainZone", 
                  DESCRIPTION="block type MainZone")
    ET.SubElement(tags, "OtherTag", ID="LT1", LABEL="DefaultLine",
                  DESCRIPTION="line type DefaultLine")
    
    # Layout
    layout = ET.SubElement(alto, "Layout")
    page = ET.SubElement(layout, "Page", ID="page1", PHYSICAL_IMG_NR="1",
                        HEIGHT="480", WIDTH="640")
    print_space = ET.SubElement(page, "PrintSpace", HEIGHT="480", WIDTH="640",
                                VPOS="0", HPOS="0")
    
    # TextBlock
    text_block = ET.SubElement(print_space, "TextBlock", ID="block_1",
                              HPOS="10", VPOS="10", WIDTH="620", HEIGHT="200",
                              TAGREFS="BT1")
    
    # TextLine 1
    text_line1 = ET.SubElement(text_block, "TextLine", ID="line_1",
                              HPOS="20", VPOS="30", WIDTH="600", HEIGHT="40",
                              TAGREFS="LT1", BASELINE="20 50 620 50")
    shape1 = ET.SubElement(text_line1, "Shape")
    ET.SubElement(shape1, "Polygon", POINTS="20,30 620,30 620,70 20,70")
    
    # TextLine 2
    text_line2 = ET.SubElement(text_block, "TextLine", ID="line_2",
                              HPOS="20", VPOS="90", WIDTH="600", HEIGHT="40",
                              TAGREFS="LT1", BASELINE="20 110 620 110")
    shape2 = ET.SubElement(text_line2, "Shape")
    ET.SubElement(shape2, "Polygon", POINTS="20,90 620,90 620,130 20,130")
    
    # Sauvegarder
    alto_path = temp_dir / "test_lines.xml"
    tree = ET.ElementTree(alto)
    tree.write(alto_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    
    return alto_path


def test_extract_lines_from_alto(sample_alto_with_lines, sample_image_path):
    """Teste l'extraction des lignes depuis un fichier ALTO."""
    image_path, lines, regions = extract_lines_from_alto(str(sample_alto_with_lines))
    
    assert image_path is not None
    assert len(lines) == 2
    assert "baseline" in lines[0]
    assert "boundary" in lines[0]
    assert "id" in lines[0]
    
    # Vérifier que la baseline a au moins 2 points
    assert len(lines[0]["baseline"]) >= 2


def test_convert_lines_to_boxes(sample_alto_with_lines):
    """Teste la conversion des lignes en boxes."""
    _, lines, _ = extract_lines_from_alto(str(sample_alto_with_lines))
    image_size = (640, 480)
    
    # Test ground truth format
    gt_boxes = convert_lines_to_boxes(lines, image_size, is_gt=True)
    assert isinstance(gt_boxes, np.ndarray)
    assert gt_boxes.shape[0] == 2
    assert gt_boxes.shape[1] == 7  # [x1, y1, x2, y2, class_id, difficult, crowd]
    
    # Test prediction format
    pred_boxes = convert_lines_to_boxes(lines, image_size, is_gt=False)
    assert pred_boxes.shape[1] == 6  # [x1, y1, x2, y2, class_id, confidence]


def test_convert_lines_to_boxes_empty():
    """Teste la conversion avec une liste vide."""
    boxes = convert_lines_to_boxes([], (640, 480), is_gt=True)
    assert boxes.shape == (0, 7)


def test_convert_lines_without_boundary():
    """Teste la conversion de lignes sans boundary (seulement baseline)."""
    lines = [{
        'id': 'line_1',
        'baseline': [[100, 50], [500, 50]],
        'boundary': None
    }]
    
    boxes = convert_lines_to_boxes(lines, (640, 480), is_gt=True)
    assert boxes.shape[0] == 1
    # Vérifie que des coordonnées valides ont été créées
    assert boxes[0][2] > boxes[0][0]  # x2 > x1
    assert boxes[0][3] > boxes[0][1]  # y2 > y1


def test_calculate_iou():
    """Teste le calcul de l'IoU entre deux boxes."""
    box1 = [0, 0, 100, 100]
    box2 = [50, 50, 150, 150]
    
    iou = calculate_iou(box1, box2)
    
    # IoU attendu: intersection = 2500, union = 17500
    expected_iou = 2500 / 17500
    assert abs(iou - expected_iou) < 0.01


def test_calculate_iou_no_overlap():
    """Teste l'IoU de boxes qui ne se chevauchent pas."""
    box1 = [0, 0, 100, 100]
    box2 = [200, 200, 300, 300]
    
    iou = calculate_iou(box1, box2)
    assert iou == 0.0


def test_calculate_iou_perfect_overlap():
    """Teste l'IoU de boxes identiques."""
    box1 = [0, 0, 100, 100]
    box2 = [0, 0, 100, 100]
    
    iou = calculate_iou(box1, box2)
    assert iou == 1.0


def test_normalize_box():
    """Teste la normalisation des coordonnées de box."""
    box = np.array([100, 50, 300, 150])
    width, height = 640, 480
    
    normalized = normalize_box(box, width, height, scale=100)
    
    assert normalized[0] == int(100 * 100 / 640)  # x1
    assert normalized[1] == int(100 * 50 / 480)   # y1
    assert normalized[2] == int(100 * 300 / 640)  # x2
    assert normalized[3] == int(100 * 150 / 480)  # y2


def test_add_line_to_element():
    """Teste l'ajout d'une ligne à un élément XML."""
    ns = "http://www.loc.gov/standards/alto/ns-v4#"
    parent = ET.Element(f"{{{ns}}}TextBlock")
    
    line_data = {
        'id': 'line_test',
        'baseline': [[100, 50], [500, 50]],
        'boundary': [[100, 30], [500, 30], [500, 70], [100, 70]]
    }
    
    line_element = _add_line_to_element(parent, line_data, tag_id="LT1")
    
    assert line_element is not None
    assert line_element.get('ID') == 'line_test'
    assert line_element.get('TAGREFS') == 'LT1'
    assert line_element.get('BASELINE') is not None
    
    # Vérifier que Shape et Polygon sont présents
    shape = line_element.find(f"{{{ns}}}Shape")
    assert shape is not None
    polygon = shape.find(f"{{{ns}}}Polygon")
    assert polygon is not None


def test_add_lines_to_alto(sample_alto_with_lines, temp_dir):
    """Teste l'ajout de nouvelles lignes à un fichier ALTO."""
    output_path = temp_dir / "output_lines.xml"
    
    # Nouvelles lignes à ajouter
    new_lines = [
        {
            'id': 'new_line_1',
            'baseline': [[30, 200], [610, 200]],
            'boundary': [[30, 180], [610, 180], [610, 220], [30, 220]]
        },
        {
            'id': 'new_line_2',
            'baseline': [[30, 250], [610, 250]],
            'boundary': [[30, 230], [610, 230], [610, 270], [30, 270]]
        }
    ]
    
    result = add_lines_to_alto(new_lines, str(output_path), str(sample_alto_with_lines))
    
    assert result is True
    assert output_path.exists()
    
    # Vérifier que les lignes ont été ajoutées
    tree = ET.parse(str(output_path))
    root = tree.getroot()
    ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
    
    text_lines = root.findall('.//alto:TextLine', ns)
    assert len(text_lines) == 2  # Les nouvelles lignes ont remplacé les anciennes


def test_add_lines_to_alto_error_handling(temp_dir):
    """Teste la gestion des erreurs lors de l'ajout de lignes."""
    non_existent = temp_dir / "non_existent.xml"
    output_path = temp_dir / "output.xml"
    
    result = add_lines_to_alto([], str(output_path), str(non_existent))
    assert result is False