"""Tests des cinq corrections du protocole de mesure de la segmentation de ligne.

Chaque classe correspond à un point du plan d'action :
  1.1  coordonnées non quantifiées pour le mAP
  1.2  propagation du score de confiance du détecteur
  1.3  parseur unique appliqué à la référence et aux prédictions
  1.4  protocole de zone symétrique entre modèles
  1.5  baseline extraite du polygone et non de sa boîte englobante
"""
import numpy as np
import pytest
from lxml import etree as ET

from src.alto.alto_lines import (
    CONFIDENCE_ATTR,
    DEFAULT_BASELINE_RATIO,
    _add_line_to_element,
    add_lines_to_alto,
    bbox_to_baseline,
    convert_lines_to_boxes,
    polygon_to_baseline,
    read_lines,
    read_lines_geometry,
)

ALTO_NS = "http://www.loc.gov/standards/alto/ns-v4#"
NS = {'alto': ALTO_NS}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_alto(blocks, page_size=(3000, 4000)):
    """Construit un ALTO minimal.

    blocks: liste de (block_id, zone_label, bbox, lines) où chaque ligne est un dict
    {'id', 'polygon', 'baseline'} — 'baseline' optionnelle.
    """
    width, height = page_size
    alto = ET.Element("alto", nsmap={None: ALTO_NS})
    desc = ET.SubElement(alto, "Description")
    ET.SubElement(desc, "MeasurementUnit").text = "pixel"
    src = ET.SubElement(desc, "sourceImageInformation")
    ET.SubElement(src, "fileName").text = "page.jpg"

    tags = ET.SubElement(alto, "Tags")
    labels = sorted({label for _, label, _, _ in blocks if label})
    tag_ids = {}
    for i, label in enumerate(labels):
        tag_ids[label] = f"BT{i}"
        ET.SubElement(tags, "OtherTag", ID=tag_ids[label], LABEL=label)
    ET.SubElement(tags, "OtherTag", ID="LT1", LABEL="DefaultLine")

    layout = ET.SubElement(alto, "Layout")
    page = ET.SubElement(layout, "Page", ID="p1", WIDTH=str(width), HEIGHT=str(height))
    print_space = ET.SubElement(page, "PrintSpace", HPOS="0", VPOS="0",
                                WIDTH=str(width), HEIGHT=str(height))

    for block_id, label, bbox, lines in blocks:
        x1, y1, x2, y2 = bbox
        attrs = dict(ID=block_id, HPOS=str(x1), VPOS=str(y1),
                     WIDTH=str(x2 - x1), HEIGHT=str(y2 - y1))
        if label:
            attrs['TAGREFS'] = tag_ids[label]
        block = ET.SubElement(print_space, "TextBlock", **attrs)
        for line in lines:
            line_attrs = dict(ID=line['id'], TAGREFS="LT1")
            if line.get('baseline'):
                line_attrs['BASELINE'] = " ".join(
                    f"{int(x)} {int(y)}" for x, y in line['baseline'])
            elem = ET.SubElement(block, "TextLine", **line_attrs)
            if line.get('polygon'):
                shape = ET.SubElement(elem, "Shape")
                ET.SubElement(shape, "Polygon", POINTS=" ".join(
                    f"{int(x)},{int(y)}" for x, y in line['polygon']))
    return alto


def _write(alto, path):
    ET.ElementTree(alto).write(str(path), xml_declaration=True, encoding="UTF-8")
    return str(path)


def _rect(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _iou(a, b):
    """IoU with the VOC pixel convention used by the mean_average_precision package."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]) + 1)
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]) + 1)
    inter = ix * iy
    area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    return inter / (area_a + area_b - inter)


def _map50(gt_lines, pred_lines, **kwargs):
    import warnings
    from mean_average_precision import MetricBuilder
    warnings.filterwarnings('ignore')
    builder = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False,
                                                    num_classes=1)
    gt = convert_lines_to_boxes(gt_lines, is_gt=True, **kwargs)
    pred = convert_lines_to_boxes(pred_lines, is_gt=False, **kwargs)
    builder.add(pred, gt)
    return builder.value(iou_thresholds=[0.5])[0.5][0]["ap"]


# ---------------------------------------------------------------------------
# 1.1 — le mAP ne doit plus être calculé sur une grille d'entiers 0-100
# ---------------------------------------------------------------------------

class TestPixelCoordinates:

    def test_boxes_are_in_pixels_by_default(self):
        lines = [{'id': 'l', 'boundary': _rect(300, 1500, 2700, 1555), 'baseline': []}]
        box = convert_lines_to_boxes(lines, is_gt=True)[0]
        assert list(box[:4]) == [300.0, 1500.0, 2700.0, 1555.0]

    def test_legacy_grid_still_reachable(self):
        """L'ancienne quantification reste disponible pour rejouer les chiffres passés."""
        lines = [{'id': 'l', 'boundary': _rect(300, 1500, 2700, 1555), 'baseline': []}]
        box = convert_lines_to_boxes(lines, image_size=(3000, 4000), is_gt=True,
                                     scale=100)[0]
        assert list(box[:4]) == [10.0, 37.0, 90.0, 38.0]

    def test_legacy_grid_collapses_a_typical_line_to_one_unit(self):
        """Une ligne de 55 px sur une page de 4000 px ne survit pas à la grille."""
        lines = [{'id': 'l', 'boundary': _rect(300, 1500, 2700, 1555), 'baseline': []}]
        box = convert_lines_to_boxes(lines, image_size=(3000, 4000), is_gt=True,
                                     scale=100)[0]
        assert box[3] - box[1] == 1

    def test_quantised_iou_depends_on_where_the_line_sits_in_the_page(self):
        """La même erreur de 15 px donne, sur la grille, un IoU qui oscille entre
        0.67 et 1.00 selon la position absolue de la ligne. En pixels, il est
        constant : c'est la qualité de la prédiction qui est mesurée, pas l'arrondi.
        """
        legacy, pixels = [], []
        for y in (1500, 1520, 1540, 1560):
            gt = [{'id': 'g', 'boundary': _rect(300, y, 2700, y + 55), 'baseline': []}]
            pred = [{'id': 'p', 'boundary': _rect(300, y + 15, 2700, y + 70),
                     'baseline': []}]
            legacy.append(_iou(
                convert_lines_to_boxes(gt, image_size=(3000, 4000), is_gt=True,
                                       scale=100)[0][:4],
                convert_lines_to_boxes(pred, image_size=(3000, 4000), is_gt=False,
                                       scale=100)[0][:4]))
            pixels.append(_iou(convert_lines_to_boxes(gt, is_gt=True)[0][:4],
                               convert_lines_to_boxes(pred, is_gt=False)[0][:4]))

        assert max(legacy) - min(legacy) > 0.3
        assert max(pixels) - min(pixels) == pytest.approx(0.0, abs=1e-9)

    def test_quantised_iou_can_erase_the_error_entirely(self):
        """À certaines positions, la grille rend la prédiction *parfaite*."""
        gt = [{'id': 'g', 'boundary': _rect(300, 1520, 2700, 1575), 'baseline': []}]
        pred = [{'id': 'p', 'boundary': _rect(300, 1535, 2700, 1590), 'baseline': []}]
        legacy = _iou(
            convert_lines_to_boxes(gt, image_size=(3000, 4000), is_gt=True,
                                   scale=100)[0][:4],
            convert_lines_to_boxes(pred, image_size=(3000, 4000), is_gt=False,
                                   scale=100)[0][:4])
        pixels = _iou(convert_lines_to_boxes(gt, is_gt=True)[0][:4],
                      convert_lines_to_boxes(pred, is_gt=False)[0][:4])
        assert legacy == pytest.approx(1.0)
        assert pixels < 0.6

    def test_pixel_iou_ranks_two_errors_of_different_size(self):
        gt = [{'id': 'g', 'boundary': _rect(300, 1500, 2700, 1555), 'baseline': []}]
        small = [{'id': 'p', 'boundary': _rect(300, 1503, 2700, 1558), 'baseline': []}]
        big = [{'id': 'p', 'boundary': _rect(300, 1520, 2700, 1575), 'baseline': []}]
        assert _map50(gt, small) > _map50(gt, big)


# ---------------------------------------------------------------------------
# 1.2 — le score de confiance du détecteur doit survivre jusqu'au scoring
# ---------------------------------------------------------------------------

class TestConfidencePropagation:

    def test_confidence_is_written_to_alto(self, temp_dir):
        parent = ET.Element(f"{{{ALTO_NS}}}TextBlock")
        elem = _add_line_to_element(
            parent, {'id': 'l', 'baseline': [[10, 50], [500, 50]],
                     'boundary': _rect(10, 30, 500, 70), 'confidence': 0.42})
        assert float(elem.get(CONFIDENCE_ATTR)) == pytest.approx(0.42)

    def test_confidence_round_trips_through_alto(self, temp_dir):
        alto = _build_alto([("b1", "MainZone", (0, 0, 3000, 4000), [])])
        path = _write(alto, temp_dir / "src.xml")
        out = str(temp_dir / "out.xml")
        lines = [{'id': f'l{i}', 'baseline': [[300, 100 + 60 * i], [2700, 100 + 60 * i]],
                  'boundary': _rect(300, 80 + 60 * i, 2700, 120 + 60 * i),
                  'confidence': 0.1 * (i + 1)} for i in range(5)]
        assert add_lines_to_alto(lines, out, path) is True

        read_back = sorted(read_lines(out), key=lambda l: l['confidence'])
        assert [round(l['confidence'], 3) for l in read_back] == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_missing_confidence_defaults_to_one(self, temp_dir):
        alto = _build_alto([("b1", "MainZone", (0, 0, 3000, 4000), [
            {'id': 'l1', 'polygon': _rect(300, 100, 2700, 155),
             'baseline': [[300, 140], [2700, 140]]}])])
        path = _write(alto, temp_dir / "gt.xml")
        assert read_lines(path)[0]['confidence'] == 1.0

    def test_confidence_reaches_the_boxes(self):
        lines = [{'id': 'l', 'boundary': _rect(0, 0, 100, 20), 'baseline': [],
                  'confidence': 0.33}]
        assert convert_lines_to_boxes(lines, is_gt=False)[0][5] == pytest.approx(0.33)

    def test_ranking_by_confidence_changes_the_ap(self):
        """L'AP intègre la courbe P/R : avec des confiances constantes, elle dégénère."""
        gt = [{'id': 'g1', 'boundary': _rect(0, 0, 1000, 60), 'baseline': []},
              {'id': 'g2', 'boundary': _rect(0, 200, 1000, 260), 'baseline': []}]
        # Un vrai positif sûr, et un faux positif que le détecteur juge douteux.
        preds = [{'id': 'bad', 'boundary': _rect(0, 1000, 1000, 1060), 'baseline': [],
                  'confidence': 0.1},
                 {'id': 'good', 'boundary': _rect(0, 0, 1000, 60), 'baseline': [],
                  'confidence': 0.9}]
        with_conf = _map50(gt, preds)

        flat = [dict(p, confidence=1.0) for p in preds]
        without_conf = _map50(gt, flat)

        # Ordonnées par confiance, le vrai positif passe avant le faux positif.
        assert with_conf > without_conf


# ---------------------------------------------------------------------------
# 1.3 — un seul parseur, appliqué identiquement à la GT et aux prédictions
# ---------------------------------------------------------------------------

class TestSingleParser:

    @pytest.fixture
    def alto_mixed_geometry(self, temp_dir):
        """Une ligne avec baseline + polygone, une avec polygone seul."""
        alto = _build_alto([("b1", "MainZone", (0, 0, 3000, 4000), [
            {'id': 'with_bl', 'polygon': _rect(300, 100, 2700, 155),
             'baseline': [[300, 140], [2700, 140]]},
            {'id': 'no_bl', 'polygon': _rect(300, 200, 2700, 255)},
        ])])
        return _write(alto, temp_dir / "mixed.xml")

    def test_polygon_only_line_is_no_longer_dropped(self, alto_mixed_geometry):
        ids = [l['id'] for l in read_lines(alto_mixed_geometry)]
        assert ids == ['with_bl', 'no_bl']

    def test_read_lines_geometry_agrees_with_read_lines(self, alto_mixed_geometry):
        """Les deux lecteurs doivent retenir le même ensemble de lignes."""
        _, geo_lines, _ = read_lines_geometry(alto_mixed_geometry)
        assert [l['id'] for l in geo_lines] == [l['id'] for l in read_lines(alto_mixed_geometry)]

    def test_same_reader_gives_identical_gt_for_both_metrics(self, alto_mixed_geometry):
        """mAP et ZoneMapAlt doivent scorer le même ensemble de lignes de référence."""
        from src.utils.zonemap import compute_zonemap_page
        lines = read_lines(alto_mixed_geometry)
        boxes = convert_lines_to_boxes(lines, is_gt=True)
        stats = compute_zonemap_page(lines, lines, with_recognition=False)
        assert boxes.shape[0] == len(lines)
        assert stats['counts']['Match'] == len(lines)
        assert stats['counts']['Miss'] == 0

    def test_degenerate_polygon_is_not_returned_as_usable_geometry(self, temp_dir):
        """Un polygone de moins de 3 points ne doit pas atteindre ZoneMapAlt."""
        alto = _build_alto([("b1", "MainZone", (0, 0, 3000, 4000), [
            {'id': 'degenerate', 'polygon': [[300, 100], [301, 100]]}])])
        path = _write(alto, temp_dir / "degenerate.xml")
        assert read_lines(path) == []

    def test_reader_is_symmetric_on_gt_and_prediction(self, temp_dir):
        """Même fichier lu en GT et en prédiction : mAP@50 = 1."""
        alto = _build_alto([("b1", "MainZone", (0, 0, 3000, 4000), [
            {'id': f'l{i}', 'polygon': _rect(300, 100 + 60 * i, 2700, 155 + 60 * i),
             'baseline': [[300, 140 + 60 * i], [2700, 140 + 60 * i]]} for i in range(6)])])
        path = _write(alto, temp_dir / "self.xml")
        lines = read_lines(path)
        assert _map50(lines, lines) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 1.4 — le protocole de zone doit être le même pour tous les modèles
# ---------------------------------------------------------------------------

class TestZoneProtocol:

    @pytest.fixture
    def layout(self, temp_dir):
        """Un layout avec une MainZone et une MarginTextZone (zone ignorée)."""
        alto = _build_alto([
            ("main", "MainZone", (300, 100, 2700, 3900), []),
            ("margin", "MarginTextZone", (2750, 100, 2950, 900), []),
        ])
        return _write(alto, temp_dir / "layout.xml")

    @staticmethod
    def _preds():
        return [
            # dans la MainZone
            {'id': 'p_main', 'boundary': _rect(400, 200, 2600, 255),
             'baseline': [[400, 240], [2600, 240]], 'confidence': 0.9},
            # dans la MarginTextZone : la GT correspondante est filtrée au scoring
            {'id': 'p_margin', 'boundary': _rect(2760, 200, 2940, 255),
             'baseline': [[2760, 240], [2940, 240]], 'confidence': 0.8},
            # hors de toute zone
            {'id': 'p_orphan', 'boundary': _rect(50, 3950, 250, 3990),
             'baseline': [[50, 3975], [250, 3975]], 'confidence': 0.3},
        ]

    def test_line_in_ignored_zone_inherits_its_tagrefs(self, layout, temp_dir):
        out = str(temp_dir / "pred.xml")
        assert add_lines_to_alto(self._preds(), out, layout) is True

        root = ET.parse(out).getroot()
        margin = [b for b in root.findall('.//alto:TextBlock', NS)
                  if b.get('ID') == 'margin']
        assert len(margin) == 1, "la zone ignorée ne doit plus être supprimée du fichier"
        assert len(margin[0].findall('alto:TextLine', NS)) == 1

    def test_line_in_ignored_zone_is_filtered_like_the_ground_truth(self, layout, temp_dir):
        out = str(temp_dir / "pred.xml")
        add_lines_to_alto(self._preds(), out, layout)
        ids = {l['id'] for l in read_lines(out)}
        # p_margin est écartée, exactement comme la ligne de référence de la marge
        assert 'p_main' in ids
        assert not any(i.startswith('p_margin') for i in ids)

    def test_orphan_becomes_a_pseudo_zone_by_default(self, layout, temp_dir):
        out = str(temp_dir / "pred.xml")
        add_lines_to_alto(self._preds(), out, layout)
        root = ET.parse(out).getroot()
        pseudo = [b for b in root.findall('.//alto:TextBlock', NS)
                  if b.get('ID', '').startswith('pseudo_block')]
        assert len(pseudo) == 1

    def test_drop_policy_constrains_predictions_to_the_layout(self, layout, temp_dir):
        """Protocole symétrique : comme Kraken, aucune ligne hors des zones."""
        out = str(temp_dir / "pred_drop.xml")
        add_lines_to_alto(self._preds(), out, layout, orphan_policy="drop")
        root = ET.parse(out).getroot()
        assert not [b for b in root.findall('.//alto:TextBlock', NS)
                    if b.get('ID', '').startswith('pseudo_block')]
        assert len(read_lines(out)) == 1  # seule p_main subsiste

    def test_unknown_orphan_policy_is_rejected(self, layout, temp_dir):
        with pytest.raises(ValueError):
            add_lines_to_alto([], str(temp_dir / "x.xml"), layout, orphan_policy="nope")

    def test_ignored_zone_does_not_leak_ground_truth_lines(self, temp_dir):
        """Les lignes de la GT présentes dans une zone ignorée ne doivent pas
        se retrouver dans le fichier de prédiction."""
        alto = _build_alto([
            ("main", "MainZone", (300, 100, 2700, 3900), []),
            ("margin", "MarginTextZone", (2750, 100, 2950, 900), [
                {'id': 'gt_margin_line', 'polygon': _rect(2760, 200, 2940, 255),
                 'baseline': [[2760, 240], [2940, 240]]}]),
        ])
        layout = _write(alto, temp_dir / "layout_gt.xml")
        out = str(temp_dir / "pred.xml")
        add_lines_to_alto(self._preds()[:1], out, layout)
        root = ET.parse(out).getroot()
        ids = {l.get('ID') for l in root.findall('.//alto:TextLine', NS)}
        assert 'gt_margin_line' not in ids

    def test_line_is_assigned_to_the_zone_that_contains_it(self, temp_dir):
        """Une ligne de MainZone qui effleure une petite zone marginale doit rester
        dans la MainZone : l'IoU, lui, favorisait la petite zone."""
        alto = _build_alto([
            ("main", "MainZone", (300, 100, 2700, 3900), []),
            ("margin", "MarginTextZone", (2650, 150, 2800, 300), []),
        ])
        layout = _write(alto, temp_dir / "overlap.xml")
        out = str(temp_dir / "pred.xml")
        line = [{'id': 'p', 'boundary': _rect(400, 200, 2700, 255),
                 'baseline': [[400, 240], [2700, 240]], 'confidence': 0.9}]
        add_lines_to_alto(line, out, layout)

        root = ET.parse(out).getroot()
        main = [b for b in root.findall('.//alto:TextBlock', NS) if b.get('ID') == 'main']
        assert len(main[0].findall('alto:TextLine', NS)) == 1

    def test_containment_ratio(self):
        from src.alto.alto_lines import calculate_containment
        assert calculate_containment([10, 10, 20, 20], [0, 0, 100, 100]) == 1.0
        assert calculate_containment([0, 0, 20, 10], [10, 0, 100, 100]) == pytest.approx(0.5)
        assert calculate_containment([0, 0, 10, 10], [50, 50, 60, 60]) == 0.0

    def test_base_line_config_selects_the_policy(self):
        from src.tasks.line.base_line import BaseLine

        class _Task(BaseLine):
            def load(self): pass
            def _process_batch(self, *a, **k): return []

        assert _Task({'device': 'cpu'}).orphan_policy == "pseudo_block"
        assert _Task({'device': 'cpu',
                      'restrict_to_layout': True}).orphan_policy == "drop"


# ---------------------------------------------------------------------------
# 1.5 — la baseline doit être extraite du polygone, pas de sa boîte englobante
# ---------------------------------------------------------------------------

class TestBaselineExtraction:

    def test_baseline_follows_a_slanted_line(self):
        """Une ligne inclinée : la baseline doit l'être aussi."""
        polygon = [[0, 0], [1000, 100], [1000, 160], [0, 60]]
        baseline = polygon_to_baseline(polygon)
        ys = [p[1] for p in baseline]
        assert ys[-1] - ys[0] == pytest.approx(100, abs=5)

    def test_baseline_follows_a_curved_line(self):
        """Une ligne ondulante : la polyligne ne doit pas être plate."""
        xs = np.linspace(0, 1000, 40)
        wave = 25 * np.sin(2 * np.pi * xs / 1000)
        top = [[x, y] for x, y in zip(xs, wave)]
        bottom = [[x, y + 60] for x, y in zip(xs[::-1], wave[::-1])]
        baseline = polygon_to_baseline(top + bottom)
        ys = np.array([p[1] for p in baseline])
        assert len(baseline) > 2
        assert ys.max() - ys.min() > 30

    def test_straight_line_collapses_to_two_points(self):
        baseline = polygon_to_baseline(_rect(0, 0, 1000, 60))
        assert len(baseline) == 2

    def test_baseline_sits_below_mid_height(self):
        """La baseline typographique est sous le milieu de la boîte, pas dessus."""
        baseline = polygon_to_baseline(_rect(0, 100, 1000, 200))
        y = baseline[0][1]
        assert y == pytest.approx(100 + DEFAULT_BASELINE_RATIO * 100, abs=1)
        assert y > 150, "l'ancienne règle plaçait la baseline à mi-hauteur"

    def test_ratio_is_configurable(self):
        assert polygon_to_baseline(_rect(0, 0, 100, 100), ratio=0.0)[0][1] == 0
        assert polygon_to_baseline(_rect(0, 0, 100, 100), ratio=1.0)[0][1] == 100

    def test_degenerate_polygons_return_none(self):
        assert polygon_to_baseline([[0, 0], [1, 1]]) is None
        assert polygon_to_baseline([[5, 0], [5, 10], [5, 20]]) is None

    def test_bbox_fallback_is_calibrated_not_centred(self):
        baseline = bbox_to_baseline(0, 100, 1000, 200)
        assert baseline == [[0, 167], [1000, 167]]

    def test_yolo_task_uses_the_polygon(self):
        """La conversion YOLO doit produire une baseline inclinée sur un masque incliné."""
        from src.tasks.line.yolo_line import YoloLineTask
        task = YoloLineTask.__new__(YoloLineTask)
        task.baseline_ratio = DEFAULT_BASELINE_RATIO

        mask = np.array([[0, 0], [1000, 100], [1000, 160], [0, 60]], dtype=float)
        line = YoloLineTask._yolo_box_to_line(
            task, [0, 0, 1000, 160], mask_xy=mask, confidence=0.7)

        ys = [p[1] for p in line['baseline']]
        assert ys[-1] > ys[0] + 50, "la baseline doit suivre l'inclinaison du masque"
        assert line['confidence'] == 0.7

    def test_yolo_detection_only_falls_back_to_calibrated_bbox(self):
        from src.tasks.line.yolo_line import YoloLineTask
        task = YoloLineTask.__new__(YoloLineTask)
        task.baseline_ratio = DEFAULT_BASELINE_RATIO

        line = YoloLineTask._yolo_box_to_line(task, [0, 100, 1000, 200], confidence=0.5)
        assert line['baseline'] == [[0, 167], [1000, 167]]
