import pytest
import tempfile
import shutil
import yaml

from pathlib import Path
from PIL import Image
from lxml import etree as ET


@pytest.fixture
def temp_dir():
    """Crée un répertoire temporaire pour les tests."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp)


@pytest.fixture
def sample_image():
    """Crée une image de test."""
    img = Image.new('RGB', (640, 480), color='white')
    return img


@pytest.fixture
def sample_image_path(temp_dir, sample_image):
    """Sauvegarde une image de test et retourne son chemin."""
    img_path = temp_dir / "test_image.jpg"
    sample_image.save(img_path)
    return img_path


@pytest.fixture
def sample_alto_xml():
    """Crée un fichier ALTO XML de test minimal."""
    ns = "http://www.loc.gov/standards/alto/ns-v4#"
    alto = ET.Element(
        "alto",
        nsmap={None: ns, "xsi": "http://www.w3.org/2001/XMLSchema-instance"}
    )
    
    # Description
    desc = ET.SubElement(alto, "Description")
    ET.SubElement(desc, "MeasurementUnit").text = "pixel"
    source = ET.SubElement(desc, "sourceImageInformation")
    ET.SubElement(source, "fileName").text = "test_image.jpg"
    
    # Tags
    tags = ET.SubElement(alto, "Tags")
    ET.SubElement(tags, "OtherTag", ID="BT1", LABEL="MainZone", 
                  DESCRIPTION="block type MainZone")
    
    # Layout
    layout = ET.SubElement(alto, "Layout")
    page = ET.SubElement(layout, "Page", ID="page1", PHYSICAL_IMG_NR="1",
                        HEIGHT="480", WIDTH="640")
    print_space = ET.SubElement(page, "PrintSpace", HEIGHT="480", WIDTH="640",
                                VPOS="0", HPOS="0")
    
    # TextBlock
    text_block = ET.SubElement(print_space, "TextBlock", ID="block_1",
                              HPOS="10", VPOS="10", WIDTH="620", HEIGHT="460",
                              TAGREFS="BT1")
    
    return alto


@pytest.fixture
def sample_alto_path(temp_dir, sample_alto_xml):
    """Sauvegarde un fichier ALTO et retourne son chemin."""
    alto_path = temp_dir / "test_alto.xml"
    tree = ET.ElementTree(sample_alto_xml)
    tree.write(alto_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    return alto_path


@pytest.fixture
def sample_config():
    """Retourne une configuration de test."""
    return {
        "run_name": "test_run",
        "output_dir": "test_results",
        "device": "cpu",
        "use_wandb": False,
        "data": {
            "test": "test_data"
        },
        "tasks": {
            "layout": {
                "type": "YoloLayout",
                "config": {
                    "model_path": "test_model.pt",
                    "batch_size": 4,
                    "img_size": 640
                }
            }
        }
    }


@pytest.fixture
def sample_config_file(temp_dir, sample_config):
    """Crée un fichier de configuration YAML."""
    config_path = temp_dir / "config.yml"
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f)
    return config_path