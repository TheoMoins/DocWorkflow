"""Tests pour la tâche HTR avec Kraken."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from lxml import etree as ET
from src.tasks.htr.kraken_htr import KrakenHTRTask


@pytest.fixture
def kraken_htr_config():
    """Configuration pour les tests Kraken HTR."""
    return {
        "model_path": "htr_model.mlmodel",
        "device": "cpu",
        "use_wandb": False
    }


@pytest.fixture
def sample_alto_with_text(temp_dir):
    """Crée un fichier ALTO avec du texte transcrit."""
    ns = "http://www.loc.gov/standards/alto/ns-v4#"
    alto = ET.Element("alto", nsmap={None: ns})
    
    # Structure basique
    desc = ET.SubElement(alto, "Description")
    ET.SubElement(desc, "MeasurementUnit").text = "pixel"
    source = ET.SubElement(desc, "sourceImageInformation")
    ET.SubElement(source, "fileName").text = "test.jpg"
    
    layout = ET.SubElement(alto, "Layout")
    page = ET.SubElement(layout, "Page", ID="p1", HEIGHT="480", WIDTH="640")
    print_space = ET.SubElement(page, "PrintSpace", HEIGHT="480", WIDTH="640")
    text_block = ET.SubElement(print_space, "TextBlock", ID="block1")
    
    # TextLine avec String
    text_line = ET.SubElement(text_block, "TextLine", ID="line1",
                              BASELINE="10 50 300 50")
    string_elem = ET.SubElement(text_line, "String")
    string_elem.set('CONTENT', 'Hello World')
    string_elem.set('WC', '0.95')
    
    # Sauvegarder
    alto_path = temp_dir / "test_text.xml"
    tree = ET.ElementTree(alto)
    tree.write(alto_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    
    return alto_path


def test_kraken_htr_initialization(kraken_htr_config):
    """Teste l'initialisation de KrakenHTRTask."""
    task = KrakenHTRTask(kraken_htr_config)
    
    assert task.name == "HTR (Kraken)"
    assert task.config == kraken_htr_config
    assert task.wandb_project == "HTR-comparison"


@patch('src.tasks.htr.kraken_htr.load_any')
@patch('src.tasks.htr.kraken_htr.os.path.exists')
def test_kraken_htr_load(mock_exists, mock_load, kraken_htr_config):
    """Teste le chargement du modèle HTR."""
    mock_exists.return_value = True
    mock_model = MagicMock()
    mock_load.return_value = mock_model
    
    task = KrakenHTRTask(kraken_htr_config)
    task.load()
    
    mock_load.assert_called_once_with(kraken_htr_config["model_path"])
    assert task.model is not None


def test_kraken_htr_load_missing_model(kraken_htr_config):
    """Teste le chargement avec un modèle manquant."""
    kraken_htr_config["model_path"] = "non_existent.mlmodel"
    task = KrakenHTRTask(kraken_htr_config)
    
    with pytest.raises(FileNotFoundError):
        task.load()


def test_kraken_htr_train_not_implemented(kraken_htr_config):
    """Teste que train n'est pas encore implémenté."""
    task = KrakenHTRTask(kraken_htr_config)
    
    # Ne devrait pas crasher
    task.train()


def test_kraken_htr_extract_text_from_alto(kraken_htr_config, sample_alto_with_text):
    """Teste l'extraction de texte depuis ALTO."""
    task = KrakenHTRTask(kraken_htr_config)
    
    lines_text = task._extract_text_from_alto(str(sample_alto_with_text))
    
    assert len(lines_text) == 1
    assert lines_text[0]['id'] == 'line1'
    assert lines_text[0]['text'] == 'Hello World'


@patch('src.tasks.htr.kraken_htr.glob.glob')
def test_kraken_htr_score_no_files(mock_glob, kraken_htr_config, temp_dir):
    """Teste score sans fichiers."""
    mock_glob.return_value = []
    
    task = KrakenHTRTask(kraken_htr_config)
    
    with pytest.raises(ValueError, match="No ground truth"):
        task.score(str(temp_dir / "pred"), str(temp_dir / "gt"))


@patch('src.tasks.htr.kraken_htr.glob.glob')
def test_kraken_htr_score_with_files(mock_glob, kraken_htr_config, 
                                     temp_dir, sample_alto_with_text):
    """Teste le scoring avec des fichiers."""
    # Créer pred et gt dirs
    pred_dir = temp_dir / "pred"
    gt_dir = temp_dir / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    
    # Copier les fichiers ALTO
    import shutil
    shutil.copy(sample_alto_with_text, pred_dir / "test.xml")
    shutil.copy(sample_alto_with_text, gt_dir / "test.xml")
    
    mock_glob.return_value = [str(gt_dir / "test.xml")]
    
    task = KrakenHTRTask(kraken_htr_config)
    
    metrics = task.score(str(pred_dir), str(gt_dir))
    
    # Devrait retourner des métriques
    assert isinstance(metrics, dict)
    assert "dataset_test/cer" in metrics
    assert "dataset_test/wer" in metrics


@patch('src.tasks.htr.kraken_htr.load_any')
@patch('src.tasks.htr.kraken_htr.os.path.exists')
@patch('src.tasks.htr.kraken_htr.glob.glob')
def test_kraken_htr_predict_no_files(mock_glob, mock_exists, mock_load,
                                     kraken_htr_config, temp_dir):
    """Teste predict sans fichiers ALTO."""
    mock_exists.return_value = True
    mock_load.return_value = MagicMock()
    mock_glob.return_value = []
    
    task = KrakenHTRTask(kraken_htr_config)
    task.load()
    
    with pytest.raises(ValueError, match="No ALTO XML files found"):
        task.predict(str(temp_dir), str(temp_dir / "output"))


def test_kraken_htr_add_text_to_alto(kraken_htr_config, sample_alto_with_text, 
                                     temp_dir):
    """Teste l'ajout de texte à un fichier ALTO."""
    task = KrakenHTRTask(kraken_htr_config)
    
    texts = [
        {'text': 'Updated text', 'confidence': 0.98}
    ]
    
    output_path = temp_dir / "output.xml"
    
    task._add_text_to_alto(
        str(sample_alto_with_text),
        texts,
        str(output_path)
    )
    
    assert output_path.exists()
    
    # Vérifier que le texte a été ajouté
    tree = ET.parse(str(output_path))
    ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
    strings = tree.findall('.//alto:String', ns)
    
    assert len(strings) == 1
    assert strings[0].get('CONTENT') == 'Updated text'