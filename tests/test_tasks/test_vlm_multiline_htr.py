"""Tests pour l'extraction d'images de la tâche HTR VLM multi-lignes."""
import pytest
from pathlib import Path
from PIL import Image
from lxml import etree as ET
from src.alto.alto_lines import read_lines_geometry
from src.tasks.htr.base_vlm_htr import bounding_box, pad_to_max_ratio, MAX_ASPECT_RATIO
from src.tasks.htr.vlm_multiline_htr import (
    VLMMultiLineHTRTask, group_lines_into_blocks, parse_multiline_output
)


@pytest.fixture
def vlm_multiline_config():
    """Configuration pour les tests VLM multi-lignes."""
    return {
        "model_name": "dummy-vlm",
        "device": "cpu",
        "use_wandb": False
    }


@pytest.fixture
def gradient_image():
    """Image dont chaque pixel encode sa position (R=x, G=y)."""
    img = Image.new('RGB', (200, 200))
    img.putdata([(x, y, 0) for y in range(200) for x in range(200)])
    return img


def test_bounding_box():
    """Teste le rectangle englobant d'un ensemble de points."""
    points = [[10, 20], [50, 5], [30, 40]]

    assert bounding_box(points) == (10, 5, 50, 40)


def test_pad_to_max_ratio_horizontal():
    """Teste le padding d'une image trop large."""
    img = Image.new('RGB', (MAX_ASPECT_RATIO * 4, 2), (255, 255, 255))

    result = pad_to_max_ratio(img)

    assert result.size == (MAX_ASPECT_RATIO * 4, 4)
    assert result.getpixel((0, 0)) == (0, 0, 0)
    assert result.getpixel((0, 1)) == (255, 255, 255)


def test_pad_to_max_ratio_unchanged():
    """Teste qu'une image au ratio acceptable n'est pas modifiée."""
    img = Image.new('RGB', (100, 50))

    assert pad_to_max_ratio(img) is img


def test_extract_multiline_image_union_bbox(vlm_multiline_config, gradient_image):
    """Teste le crop sur l'union des polygones, sans masque polygonal."""
    task = VLMMultiLineHTRTask(vlm_multiline_config)
    # Deux lignes triangulaires disjointes
    boundaries = [
        [[10, 20], [60, 20], [10, 30]],
        [[40, 50], [90, 50], [90, 70]],
    ]

    result = task._extract_multiline_image(gradient_image, boundaries)

    assert result.size == (80, 50)
    # Coins hors de tout polygone : pixels d'origine conservés (pas de masque noir)
    assert result.getpixel((79, 0)) == (89, 20, 0)
    assert result.getpixel((0, 49)) == (10, 69, 0)
    # Le crop de ligne unique, lui, masque ce même coin
    line = task._extract_line_image(gradient_image, boundaries[0])
    assert line.getpixel((49, 9)) == (0, 0, 0)


def test_extract_multiline_image_empty(vlm_multiline_config, gradient_image):
    """Teste l'extraction sans polygone exploitable."""
    task = VLMMultiLineHTRTask(vlm_multiline_config)

    assert task._extract_multiline_image(gradient_image, []) is None
    assert task._extract_multiline_image(gradient_image, [[], None]) is None


@pytest.fixture
def alto_two_zones(temp_dir):
    """Crée un ALTO avec deux TextBlocks MainZone : 3 lignes puis 2 lignes."""
    ns = "http://www.loc.gov/standards/alto/ns-v4#"
    alto = ET.Element("alto", nsmap={None: ns})

    desc = ET.SubElement(alto, "Description")
    ET.SubElement(desc, "MeasurementUnit").text = "pixel"
    source = ET.SubElement(desc, "sourceImageInformation")
    ET.SubElement(source, "fileName").text = "test.jpg"

    tags = ET.SubElement(alto, "Tags")
    ET.SubElement(tags, "OtherTag", ID="BT1", LABEL="MainZone")

    layout = ET.SubElement(alto, "Layout")
    page = ET.SubElement(layout, "Page", ID="p1", HEIGHT="480", WIDTH="640")
    print_space = ET.SubElement(page, "PrintSpace", HEIGHT="480", WIDTH="640")

    # Deux zones du même type : seule l'identité du TextBlock les distingue
    for block_id, x0, n_lines in [("block1", 10, 3), ("block2", 330, 2)]:
        text_block = ET.SubElement(print_space, "TextBlock", ID=block_id, TAGREFS="BT1")
        for i in range(n_lines):
            y = 50 + 40 * i
            ET.SubElement(text_block, "TextLine", ID=f"{block_id}_l{i}",
                          BASELINE=f"{x0} {y} {x0 + 280} {y}")

    alto_path = temp_dir / "test_two_zones.xml"
    tree = ET.ElementTree(alto)
    tree.write(alto_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")

    return alto_path


def test_group_lines_into_blocks_respects_zones(alto_two_zones):
    """Teste qu'aucun bloc ne chevauche deux zones et que le dernier bloc peut être < k."""
    _, lines, _ = read_lines_geometry(str(alto_two_zones))
    assert [line["block_id"] for line in lines] == ["block1"] * 3 + ["block2"] * 2

    blocks = group_lines_into_blocks(lines, k=2)

    assert [[line["id"] for line in block] for block in blocks] == [
        ["block1_l0", "block1_l1"],
        ["block1_l2"],
        ["block2_l0", "block2_l1"],
    ]
    for block in blocks:
        assert len({line["block_id"] for line in block}) == 1


def test_group_lines_into_blocks_k_values(alto_two_zones):
    """Teste k=1 (ligne-à-ligne) et k plus grand que les zones."""
    _, lines, _ = read_lines_geometry(str(alto_two_zones))

    assert [len(b) for b in group_lines_into_blocks(lines, k=1)] == [1, 1, 1, 1, 1]
    assert [len(b) for b in group_lines_into_blocks(lines, k=8)] == [3, 2]
    assert group_lines_into_blocks([], k=4) == []
    with pytest.raises(ValueError):
        group_lines_into_blocks(lines, k=0)


def test_lines_per_crop_config(vlm_multiline_config):
    """Teste le paramètre lines_per_crop (défaut 4)."""
    assert VLMMultiLineHTRTask(vlm_multiline_config).lines_per_crop == 4

    vlm_multiline_config["lines_per_crop"] = 2
    assert VLMMultiLineHTRTask(vlm_multiline_config).lines_per_crop == 2


def test_parse_multiline_output_nominal():
    """Teste le cas nominal : autant de lignes que k."""
    texts, n_mismatch, n_deduped = parse_multiline_output("alpha\nbeta\ngamma", k=3)

    assert texts == ["alpha", "beta", "gamma"]
    assert (n_mismatch, n_deduped) == (0, 0)


def test_parse_multiline_output_too_short():
    """Teste le repli quand il y a moins de lignes que k : complété par des chaînes vides."""
    texts, n_mismatch, n_deduped = parse_multiline_output("alpha\nbeta", k=4)

    assert texts == ["alpha", "beta", "", ""]
    assert (n_mismatch, n_deduped) == (1, 0)


def test_parse_multiline_output_too_long():
    """Teste le repli quand il y a plus de lignes que k : tronqué aux k premières."""
    texts, n_mismatch, n_deduped = parse_multiline_output("alpha\nbeta\ngamma\ndelta", k=2)

    assert texts == ["alpha", "beta"]
    assert (n_mismatch, n_deduped) == (1, 0)


def test_parse_multiline_output_dedup_absorbs_surplus():
    """Teste qu'un doublon consécutif quasi identique est collapsé avant l'alignement sur k."""
    raw = "la belle maison\nla belle maison.\nlle dernier mot"
    texts, n_mismatch, n_deduped = parse_multiline_output(raw, k=2)

    assert texts == ["la belle maison", "lle dernier mot"]
    assert (n_mismatch, n_deduped) == (0, 1)


def test_parse_multiline_output_dedup_threshold():
    """Teste que des lignes distinctes (distance au-dessus du seuil) ne sont pas collapsées."""
    raw = "la belle maison\nla belle maisson rouge"
    texts, n_mismatch, n_deduped = parse_multiline_output(raw, k=2)

    assert texts == ["la belle maison", "la belle maisson rouge"]
    assert (n_mismatch, n_deduped) == (0, 0)

    # Un seuil plus large collapse la même paire
    _, _, n_deduped = parse_multiline_output(raw, k=2, dedup_threshold=0.5)
    assert n_deduped == 1


def test_parse_multiline_output_empty_lines_not_deduped():
    """Teste que deux lignes vides consécutives ne sont pas traitées comme un doublon."""
    texts, _, n_deduped = parse_multiline_output("alpha\n\n\nbeta", k=4)

    assert texts == ["alpha", "", "", "beta"]
    assert n_deduped == 0


def test_recognize_batch_counts_and_none_crop(vlm_multiline_config, sample_image):
    """Teste _recognize_batch (modèle mocké) : un résultat par bloc, taille = k réel du bloc."""
    task = VLMMultiLineHTRTask(vlm_multiline_config)
    task.is_minicpm = True
    task._generate_from_messages = lambda messages: "alpha\nbeta"

    results = task._recognize_batch([(sample_image, 3), (None, 2)])

    assert [t["text"] for t in results[0]["texts"]] == ["alpha", "beta", ""]
    assert results[0]["n_mismatch"] == 1
    assert [t["text"] for t in results[1]["texts"]] == ["", ""]
    assert results[1]["n_mismatch"] == 0


@pytest.fixture
def alto_page_with_ignored_zone(temp_dir):
    """
    Page ALTO + image : MainZone (3 lignes), MarginTextZone ignorée (1 ligne), MainZone (2 lignes).
    La ligne de marge est absente de read_lines_geometry mais présente dans l'ALTO de sortie.
    """
    src_dir = temp_dir / "src"
    src_dir.mkdir()
    Image.new('RGB', (640, 480), color='white').save(src_dir / "page.jpg")

    ns = "http://www.loc.gov/standards/alto/ns-v4#"
    alto = ET.Element("alto", nsmap={None: ns})
    desc = ET.SubElement(alto, "Description")
    ET.SubElement(desc, "MeasurementUnit").text = "pixel"
    source = ET.SubElement(desc, "sourceImageInformation")
    ET.SubElement(source, "fileName").text = "page.jpg"
    tags = ET.SubElement(alto, "Tags")
    ET.SubElement(tags, "OtherTag", ID="BT1", LABEL="MainZone")
    ET.SubElement(tags, "OtherTag", ID="BT2", LABEL="MarginTextZone")
    layout = ET.SubElement(alto, "Layout")
    page = ET.SubElement(layout, "Page", ID="p1", HEIGHT="480", WIDTH="640")
    print_space = ET.SubElement(page, "PrintSpace", HEIGHT="480", WIDTH="640")

    for block_id, tag, n_lines in [("b1", "BT1", 3), ("margin", "BT2", 1), ("b2", "BT1", 2)]:
        text_block = ET.SubElement(print_space, "TextBlock", ID=block_id, TAGREFS=tag)
        for i in range(n_lines):
            y = 50 + 40 * i
            line = ET.SubElement(text_block, "TextLine", ID=f"{block_id}_l{i}",
                                 BASELINE=f"10 {y} 290 {y}")
            shape = ET.SubElement(line, "Shape")
            ET.SubElement(shape, "Polygon", POINTS=f"10 {y - 20} 290 {y - 20} 290 {y + 5} 10 {y + 5}")
            ET.SubElement(line, "String", CONTENT="GT text", WC="1.0")

    alto_path = src_dir / "page.xml"
    ET.ElementTree(alto).write(alto_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    return alto_path


def _read_line_texts(alto_path):
    ns = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}
    root = ET.parse(str(alto_path)).getroot()
    return {
        line.get("ID"): " ".join(s.get("CONTENT", "") for s in line.findall("alto:String", ns))
        for line in root.findall(".//alto:TextLine", ns)
    }


def test_process_batch_writes_alto_by_line_id(vlm_multiline_config, alto_page_with_ignored_zone, temp_dir):
    """
    Test d'intégration (VLM mocké) : un texte par TextLine, dans l'ordre, réattribué par ID,
    y compris avec un bloc trop long et un bloc trop court (repli 0.3).
    """
    vlm_multiline_config["lines_per_crop"] = 2
    task = VLMMultiLineHTRTask(vlm_multiline_config)
    task.is_minicpm = True
    # Blocs attendus : [b1_l0, b1_l1], [b1_l2], [b2_l0, b2_l1]
    outputs = iter(["a0\na1", "a2\nsurplus", "b0"])
    task._generate_from_messages = lambda messages: next(outputs)

    out_dir = temp_dir / "out"
    out_dir.mkdir()
    results = task._process_batch(
        [str(alto_page_with_ignored_zone)], str(alto_page_with_ignored_zone.parent), str(out_dir)
    )

    assert len(results) == 1
    assert results[0]["n_blocks"] == 3
    assert results[0]["n_mismatch"] == 2
    assert results[0]["n_deduped"] == 0
    assert _read_line_texts(out_dir / "page.xml") == {
        "b1_l0": "a0",
        "b1_l1": "a1",
        "b1_l2": "a2",
        "margin_l0": "",  # zone ignorée : pas de texte, pas de décalage des suivantes
        "b2_l0": "b0",
        "b2_l1": "",      # ligne manquante du bloc court : complétée vide
    }
    assert (out_dir / "page.jpg").exists()


def test_process_batch_resolves_conventions(vlm_multiline_config, alto_page_with_ignored_zone, temp_dir):
    """Teste que {conventions} du prompt_template est résolu depuis metadata.json du document."""
    import json
    (alto_page_with_ignored_zone.parent / "metadata.json").write_text(
        json.dumps({"conventions": {"abbreviations": "keep"}})
    )
    vlm_multiline_config["prompt_template"] = "Transcribe.\n{conventions}"
    task = VLMMultiLineHTRTask(vlm_multiline_config)
    task.is_minicpm = True
    prompts = []

    def fake_generate(messages):
        prompts.append(messages[0]["content"][0]["text"])
        return "x"

    task._generate_from_messages = fake_generate
    out_dir = temp_dir / "out"
    out_dir.mkdir()
    task._process_batch([str(alto_page_with_ignored_zone)], str(alto_page_with_ignored_zone.parent),
                        str(out_dir), save_image=False)

    assert prompts
    assert all(p == "Transcribe.\nKeep abbreviations as written, do not expand them." for p in prompts)


def test_multiline_config_params(vlm_multiline_config):
    """Teste les paramètres de config 0.5 et leurs défauts."""
    task = VLMMultiLineHTRTask(vlm_multiline_config)
    assert (task.lines_per_crop, task.rolling_window, task.dedup_threshold) == (4, False, 0.1)

    vlm_multiline_config.update({"rolling_window": True, "dedup_threshold": 0.3})
    task = VLMMultiLineHTRTask(vlm_multiline_config)
    assert (task.rolling_window, task.dedup_threshold) == (True, 0.3)


def _write_training_page(directory, texts, name="page"):
    """Écrit un ALTO à un TextBlock (une ligne par texte, avec polygone) et son image."""
    directory.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (640, 480), color='white').save(directory / f"{name}.jpg")

    ns = "http://www.loc.gov/standards/alto/ns-v4#"
    alto = ET.Element("alto", nsmap={None: ns})
    desc = ET.SubElement(alto, "Description")
    ET.SubElement(desc, "MeasurementUnit").text = "pixel"
    source = ET.SubElement(desc, "sourceImageInformation")
    ET.SubElement(source, "fileName").text = f"{name}.jpg"
    tags = ET.SubElement(alto, "Tags")
    ET.SubElement(tags, "OtherTag", ID="BT1", LABEL="MainZone")
    layout = ET.SubElement(alto, "Layout")
    page = ET.SubElement(layout, "Page", ID="p1", HEIGHT="480", WIDTH="640")
    print_space = ET.SubElement(page, "PrintSpace", HEIGHT="480", WIDTH="640")
    text_block = ET.SubElement(print_space, "TextBlock", ID="b1", TAGREFS="BT1")
    for i, text in enumerate(texts):
        y = 40 + 60 * i
        line = ET.SubElement(text_block, "TextLine", ID=f"l{i}", BASELINE=f"10 {y} 290 {y}")
        shape = ET.SubElement(line, "Shape")
        ET.SubElement(shape, "Polygon", POINTS=f"10 {y - 20} 290 {y - 20} 290 {y + 5} 10 {y + 5}")
        ET.SubElement(line, "String", CONTENT=text, WC="1.0")

    ET.ElementTree(alto).write(directory / f"{name}.xml", pretty_print=True,
                               xml_declaration=True, encoding="UTF-8")


def test_prepare_training_data_blocks_of_k(vlm_multiline_config, temp_dir):
    """6 lignes, k=4 : un bloc de 4 puis un bloc de 2, textes dans l'ordre de lecture."""
    _write_training_page(temp_dir / "train", [f"ligne {i}" for i in range(6)])
    task = VLMMultiLineHTRTask(vlm_multiline_config)

    samples = task._prepare_training_data_lines(temp_dir / "train")

    assert [s["text"] for s in samples] == [
        ["ligne 0", "ligne 1", "ligne 2", "ligne 3"],
        ["ligne 4", "ligne 5"],
    ]
    assert [len(s["boundary"]) for s in samples] == [4, 2]


def test_prepare_training_data_empty_gt_closes_block(vlm_multiline_config, temp_dir):
    """Une ligne sans GT est exclue et clôt le bloc courant (pas de ligne visible absente de la cible)."""
    _write_training_page(temp_dir / "train", ["a", "b", "", "c", "d", "e"])
    task = VLMMultiLineHTRTask(vlm_multiline_config)

    samples = task._prepare_training_data_lines(temp_dir / "train")

    assert [s["text"] for s in samples] == [["a", "b"], ["c", "d", "e"]]


def test_format_conversation_single_crop(vlm_multiline_config, temp_dir):
    """Un seul crop rectangle englobant en entrée, les k textes joints par '\\n' en sortie."""
    _write_training_page(temp_dir / "train", ["alpha", "beta", "gamma"])
    vlm_multiline_config["lines_per_crop"] = 3
    task = VLMMultiLineHTRTask(vlm_multiline_config)
    sample = task._prepare_training_data_lines(temp_dir / "train")[0]

    conv = task._format_conversation(sample, lambda path: Image.open(path).convert("RGB"))

    user, assistant = conv["messages"]
    images = [c["image"] for c in user["content"] if c["type"] == "image"]
    assert len(images) == 1
    # Union des polygones : x 10..290, y 20..165
    assert images[0].size == (280, 145)
    assert assistant["content"] == [{"type": "text", "text": "alpha\nbeta\ngamma"}]


def test_prepare_samples_with_conventions_multiline(vlm_multiline_config, temp_dir):
    """Le prompt d'entraînement résout {conventions} par document, comme VLMLineHTR."""
    import json
    _write_training_page(temp_dir / "train" / "doc_a", ["a", "b"])
    _write_training_page(temp_dir / "train" / "doc_b", ["c", "d"])
    (temp_dir / "train" / "doc_a" / "metadata.json").write_text(
        json.dumps({"conventions": {"abbreviations": "keep"}})
    )
    vlm_multiline_config["lines_per_crop"] = 2
    task = VLMMultiLineHTRTask(vlm_multiline_config)

    samples = task._prepare_samples_with_conventions(temp_dir / "train", "Transcribe.\n{conventions}")

    prompts = {tuple(s["text"]): s["prompt"] for s in samples}
    assert prompts[("a", "b")] == "Transcribe.\nKeep abbreviations as written, do not expand them."
    assert prompts[("c", "d")] == "Transcribe."


def test_prepare_training_data_skips_pages_shorter_than_k(vlm_multiline_config, temp_dir):
    """Une page avec moins de k lignes utilisables (ex. une ligne par ALTO) est ignorée."""
    _write_training_page(temp_dir / "train", ["seule ligne"], name="single")
    _write_training_page(temp_dir / "train", ["a", "", "b", "c"], name="sparse")
    _write_training_page(temp_dir / "train", ["a", "b", "c", "d", "e"], name="full")
    task = VLMMultiLineHTRTask(vlm_multiline_config)

    samples = task._prepare_training_data_lines(temp_dir / "train")

    assert {Path(s["page_image_path"]).stem for s in samples} == {"full"}
