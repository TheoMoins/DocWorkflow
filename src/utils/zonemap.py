"""
ZoneMapAlt and ZoneMapAltCnt metrics for segmentation evaluation.

ZoneMapAlt    → detection only       (Line task)
ZoneMapAltCnt → detection + content  (HTR task)

References:
  ZoneMap       https://bit.ly/2QSE3on
  ZoneMapAlt    https://bit.ly/389ruuF
  ZoneMapAltCnt https://bit.ly/2NqvSxo
"""

import copy
from operator import itemgetter
from typing import Dict, List, Optional

from shapely.geometry import Polygon, box as shapely_box
from shapely.validation import make_valid

BETA = 0.2  # minimum coverage ratio to accept a link
MS = 0.5    # split/merge penalty coefficient

_CONFIGS = ('Match', 'Miss', 'FA', 'Split', 'Merge', 'Multiple')


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------

class _Zone:
    def __init__(self, zone_id: int, polygon: Polygon, text: str = ""):
        self.id = zone_id
        self.polygon = polygon
        self.text = text
        self.linked_zones: List[int] = []


class _ZoneGroup:
    @staticmethod
    def assign_config(gt_card: int, dt_card: int) -> str:
        if gt_card > 0 and dt_card == 0: return 'Miss'
        if gt_card == 0 and dt_card > 0: return 'FA'
        if gt_card == 1 and dt_card == 1: return 'Match'
        if gt_card == 1 and dt_card > 1:  return 'Split'
        if gt_card > 1 and dt_card == 1:  return 'Merge'
        return 'Multiple'

    def __init__(self, polygon: Polygon, gt_zone=None, dt_zone=None, config=None):
        self.polygon = polygon
        self.gt_zone = gt_zone
        self.dt_zone = dt_zone
        if dt_zone is None:
            self.gt_card, self.dt_card = 1, 0
        elif gt_zone is None:
            self.gt_card, self.dt_card = 0, 1
        else:
            self.gt_card = len(dt_zone.linked_zones)
            self.dt_card = len(gt_zone.linked_zones)
        self.config = config or _ZoneGroup.assign_config(self.gt_card, self.dt_card)


# ---------------------------------------------------------------------------
# Zone construction from read_lines_geometry() dicts
# ---------------------------------------------------------------------------

def _polygon_from_line(line: dict) -> Polygon:
    """Build Shapely polygon, preferring boundary polygon over baseline bbox.

    Applies make_valid() to handle self-intersecting ALTO polygons.
    """
    if line.get('boundary') and len(line['boundary']) >= 3:
        poly = Polygon(line['boundary'])
        if not poly.is_valid:
            poly = make_valid(poly)
        return poly
    pts = line.get('baseline', [])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    pad = 10
    return shapely_box(min(xs), min(ys) - pad, max(xs), max(ys) + pad)


def _build_zones(lines: list) -> Dict[int, _Zone]:
    return {
        i: _Zone(i, _polygon_from_line(line), line.get('text', ''))
        for i, line in enumerate(lines)
    }


# ---------------------------------------------------------------------------
# ZoneMapAlt core — spatial grouping
# ---------------------------------------------------------------------------

def _ensure_valid(geom: Polygon) -> Polygon:
    """Return a valid geometry, repairing it if necessary."""
    if not geom.is_valid:
        return make_valid(geom)
    return geom


def _safe_difference(a: Polygon, b: Polygon) -> Polygon:
    return _ensure_valid(a.difference(b))


def _safe_intersection(a: Polygon, b: Polygon) -> Polygon:
    return _ensure_valid(a.intersection(b))


def _get_link_strength(poly_a: Polygon, poly_b: Polygon) -> float:
    try:
        inter = _safe_intersection(poly_a, poly_b)
        if inter.is_empty or inter.area == 0:
            return 0.0
        return (inter.area / poly_a.area) ** 2 + (inter.area / poly_b.area) ** 2
    except Exception:
        return 0.0


def _build_link_table(gt_zones: Dict[int, _Zone], dt_zones: Dict[int, _Zone]) -> list:
    table = []
    for gi, gz in gt_zones.items():
        for di, dz in dt_zones.items():
            s = _get_link_strength(gz.polygon, dz.polygon)
            if s > 0:
                table.append((gi, di, s))
    return sorted(table, key=itemgetter(2), reverse=True)


def _group_linked(link_table: list, gt_zones: Dict[int, _Zone],
                  dt_zones: Dict[int, _Zone]) -> List[_ZoneGroup]:
    groups = []
    for gi, di, _ in link_table:
        try:
            gz = gt_zones[gi]
            dz = dt_zones[di]
            temp_gt = copy.copy(gz.polygon)
            temp_dt = copy.copy(dz.polygon)

            for used_gi in dz.linked_zones:
                used_poly = gt_zones[used_gi].polygon
                temp_gt = _safe_difference(temp_gt, _safe_intersection(temp_gt, used_poly))
                temp_dt = _safe_difference(temp_dt, _safe_intersection(temp_dt, used_poly))

            for used_di in gz.linked_zones:
                used_poly = dt_zones[used_di].polygon
                temp_gt = _safe_difference(temp_gt, _safe_intersection(temp_gt, used_poly))

            remaining = _safe_intersection(temp_gt, temp_dt)
            if remaining.is_empty:
                continue
            coverage = remaining.area / temp_gt.area if temp_gt.area > 0 else 0.0
            if coverage > BETA:
                gz.linked_zones.append(di)
                dz.linked_zones.append(gi)
                groups.append(_ZoneGroup(remaining, gt_zone=gz, dt_zone=dz))
        except Exception:
            continue

    return groups


def _group_non_linked(groups: List[_ZoneGroup], gt_zones: Dict[int, _Zone],
                      dt_zones: Dict[int, _Zone]) -> List[_ZoneGroup]:
    for gz in gt_zones.values():
        if not gz.text.strip():
            continue
        try:
            temp = copy.copy(gz.polygon)
            for di in gz.linked_zones:
                temp = _safe_difference(temp, _safe_intersection(temp, dt_zones[di].polygon))
            if temp.area > 0:
                groups.append(_ZoneGroup(temp, gt_zone=gz, config='Miss'))
        except Exception:
            continue

    for dz in dt_zones.values():
        if not dz.text.strip():
            continue
        try:
            temp = copy.copy(dz.polygon)
            for gi in dz.linked_zones:
                temp = _safe_difference(temp, _safe_intersection(temp, gt_zones[gi].polygon))
            if temp.area > 0:
                groups.append(_ZoneGroup(temp, dt_zone=dz, config='FA'))
        except Exception:
            continue

    return groups


# ---------------------------------------------------------------------------
# Detection scoring
# ---------------------------------------------------------------------------

def _calc_detection_raw(groups: List[_ZoneGroup]) -> tuple:
    areas = {c: 0.0 for c in _CONFIGS}
    counts = {c: 0 for c in _CONFIGS}
    for g in groups:
        c = g.config
        area = g.polygon.area
        # Only count a Miss/FA if the source zone is completely unlinked (no Match/Split/Merge).
        # Residuals of partially-overlapping linked zones contribute to areas but not to counts.
        if c == 'Miss' and g.gt_zone and g.gt_zone.linked_zones:
            pass
        elif c == 'FA' and g.dt_zone and g.dt_zone.linked_zones:
            pass
        else:
            counts[c] += 1
        if c in ('Match', 'Miss', 'FA'):
            areas[c] += area
        elif c == 'Split':
            areas[c] += area * MS * g.dt_card
        elif c == 'Merge':
            areas[c] += area * MS * g.gt_card
        elif c == 'Multiple':
            areas[c] += area * MS * (g.gt_card + g.dt_card)
    return areas, counts


# ---------------------------------------------------------------------------
# Recognition scoring (ZoneMapAltCnt)
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]


def _arrange_by_pos(zone_ids: list, zones: Dict[int, _Zone]) -> List[str]:
    sorted_zones = sorted(
        (zones[i] for i in zone_ids),
        key=lambda z: z.polygon.bounds[1] * 1e6 + z.polygon.bounds[0]
    )
    return [z.text for z in sorted_zones]


def _word_token_matches(gt_text: str, dt_text: str) -> int:
    """Count matching word tokens (bag-of-words, case-sensitive)."""
    dt_tokens = list(dt_text.split())
    count = 0
    for w in gt_text.split():
        if w in dt_tokens:
            count += 1
            dt_tokens.remove(w)
    return count


def _calc_recognition_raw(groups: List[_ZoneGroup], gt_zones: Dict[int, _Zone],
                           dt_zones: Dict[int, _Zone]) -> dict:
    used_gt: set = set()
    used_dt: set = set()
    gt_chars = dt_chars = char_edits = 0
    gt_words = dt_words = correct_words = 0

    by_config: Dict[str, List[_ZoneGroup]] = {c: [] for c in _CONFIGS}
    for g in groups:
        by_config[g.config].append(g)

    for g in by_config['Multiple']:
        dt_ids = [i for i in g.gt_zone.linked_zones if i not in used_dt]
        gt_ids = [i for i in g.dt_zone.linked_zones if i not in used_gt]
        used_dt.update(dt_ids)
        used_gt.update(gt_ids)
        dt_texts = _arrange_by_pos(dt_ids, dt_zones)
        gt_texts = _arrange_by_pos(gt_ids, gt_zones)
        dt_str, gt_str = ''.join(dt_texts), ''.join(gt_texts)
        char_edits += _edit_distance(dt_str, gt_str)
        gt_chars += len(gt_str); dt_chars += len(dt_str)
        gt_words += len(gt_str.split()); dt_words += len(dt_str.split())
        correct_words += _word_token_matches(gt_str, dt_str)

    for g in by_config['Split']:
        if g.gt_zone.id in used_gt:
            continue
        used_gt.add(g.gt_zone.id)
        dt_ids = [i for i in g.gt_zone.linked_zones if i not in used_dt]
        used_dt.update(dt_ids)
        dt_texts = _arrange_by_pos(dt_ids, dt_zones)
        gt_text = g.gt_zone.text
        dt_str = ''.join(dt_texts)
        char_edits += _edit_distance(dt_str, gt_text)
        gt_chars += len(gt_text); dt_chars += len(dt_str)
        gt_words += len(gt_text.split()); dt_words += len(dt_str.split())
        correct_words += _word_token_matches(gt_text, dt_str)

    for g in by_config['Merge']:
        if g.dt_zone.id in used_dt:
            continue
        used_dt.add(g.dt_zone.id)
        gt_ids = [i for i in g.dt_zone.linked_zones if i not in used_gt]
        used_gt.update(gt_ids)
        gt_texts = _arrange_by_pos(gt_ids, gt_zones)
        dt_text = g.dt_zone.text
        gt_str = ''.join(gt_texts)
        char_edits += _edit_distance(dt_text, gt_str)
        gt_chars += len(gt_str); dt_chars += len(dt_text)
        gt_words += len(gt_str.split()); dt_words += len(dt_text.split())
        correct_words += _word_token_matches(gt_str, dt_text)

    for g in by_config['Match']:
        if g.gt_zone.id in used_gt or g.dt_zone.id in used_dt:
            continue
        used_gt.add(g.gt_zone.id)
        used_dt.add(g.dt_zone.id)
        gt_text, dt_text = g.gt_zone.text, g.dt_zone.text
        char_edits += _edit_distance(dt_text, gt_text)
        gt_chars += len(gt_text); dt_chars += len(dt_text)
        gt_words += len(gt_text.split()); dt_words += len(dt_text.split())
        correct_words += _word_token_matches(gt_text, dt_text)

    for g in by_config['Miss']:
        if g.gt_zone.id in used_gt:
            continue
        used_gt.add(g.gt_zone.id)
        gt_chars += len(g.gt_zone.text)
        char_edits += len(g.gt_zone.text)
        gt_words += len(g.gt_zone.text.split())

    for g in by_config['FA']:
        if g.dt_zone.id in used_dt:
            continue
        used_dt.add(g.dt_zone.id)
        dt_chars += len(g.dt_zone.text)
        char_edits += len(g.dt_zone.text)
        dt_words += len(g.dt_zone.text.split())

    return {
        'gt_chars': gt_chars, 'dt_chars': dt_chars, 'char_edits': char_edits,
        'gt_words': gt_words, 'dt_words': dt_words, 'correct_words': correct_words,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _empty_stats() -> dict:
    return {
        'total_gt_area': 0.0,
        'areas': {c: 0.0 for c in _CONFIGS},
        'counts': {c: 0 for c in _CONFIGS},
    }


def compute_zonemap_page(gt_lines: list, dt_lines: list,
                         with_recognition: bool = False) -> dict:
    """
    Compute raw ZoneMapAlt/AltCnt stats for a single page.

    Args:
        gt_lines: line dicts from read_lines_geometry() — need 'baseline' and
                  optionally 'boundary'; for recognition also 'text'.
        dt_lines: same format for predictions.
        with_recognition: if True, compute char/word error stats (ZoneMapAltCnt).

    Returns:
        Raw stats dict to be passed to accumulate_zonemap_stats().
    """
    stats = _empty_stats()
    if not gt_lines:
        return stats

    gt_zones = _build_zones(gt_lines)
    stats['total_gt_area'] = sum(z.polygon.area for z in gt_zones.values())

    if not dt_lines:
        for gz in gt_zones.values():
            stats['areas']['Miss'] += gz.polygon.area
            stats['counts']['Miss'] += 1
        if with_recognition:
            stats['recognition'] = {
                'gt_chars': 0, 'dt_chars': 0, 'char_edits': 0,
                'gt_words': 0, 'dt_words': 0, 'correct_words': 0,
            }
        return stats

    dt_zones = _build_zones(dt_lines)
    link_table = _build_link_table(gt_zones, dt_zones)
    groups = _group_linked(link_table, gt_zones, dt_zones)
    groups = _group_non_linked(groups, gt_zones, dt_zones)

    areas, counts = _calc_detection_raw(groups)
    stats['areas'] = areas
    stats['counts'] = counts

    if with_recognition:
        stats['recognition'] = _calc_recognition_raw(groups, gt_zones, dt_zones)

    return stats


def accumulate_zonemap_stats(accumulated: Optional[dict], page_stats: dict) -> dict:
    """Merge page_stats into accumulated totals."""
    if accumulated is None:
        accumulated = _empty_stats()

    accumulated['total_gt_area'] += page_stats['total_gt_area']
    for c in _CONFIGS:
        accumulated['areas'][c] += page_stats['areas'][c]
        accumulated['counts'][c] += page_stats['counts'][c]

    if 'recognition' in page_stats:
        if 'recognition' not in accumulated:
            accumulated['recognition'] = {
                'gt_chars': 0, 'dt_chars': 0, 'char_edits': 0,
                'gt_words': 0, 'dt_words': 0, 'correct_words': 0,
            }
        for k, v in page_stats['recognition'].items():
            accumulated['recognition'][k] += v

    return accumulated


def finalize_zonemap_metrics(accumulated: dict) -> dict:
    """
    Convert accumulated raw stats into the final metrics dict.

    zonemap/score          — detection error % (lower is better; >100 means many FA)
    zonemap/{config}       — area contribution per category, normalized by total GT area
    zonemap/n_{config}     — zone-group count per category
    zonemap/char_precision — char precision accounting for segmentation (HTR only)
    zonemap/char_recall    — char recall (HTR only)
    zonemap/word_precision — word precision (HTR only)
    zonemap/word_recall    — word recall (HTR only)
    """
    total = accumulated['total_gt_area']
    areas = accumulated['areas']
    counts = accumulated['counts']

    error_area = (areas['Miss'] + areas['FA'] +
                  areas['Split'] + areas['Merge'] + areas['Multiple'])
    score = round(error_area / total * 100, 4) if total > 0 else 0.0

    def _norm(v: float) -> float:
        return round(v / total, 4) if total > 0 else 0.0

    metrics = {
        'zonemap/score':         score,
        'zonemap/match':         _norm(areas['Match']),
        'zonemap/miss':          _norm(areas['Miss']),
        'zonemap/false_alarm':   _norm(areas['FA']),
        'zonemap/split':         _norm(areas['Split']),
        'zonemap/merge':         _norm(areas['Merge']),
        'zonemap/multiple':      _norm(areas['Multiple']),
        'zonemap/n_match':       counts['Match'],
        'zonemap/n_miss':        counts['Miss'],
        'zonemap/n_false_alarm': counts['FA'],
        'zonemap/n_split':       counts['Split'],
        'zonemap/n_merge':       counts['Merge'],
        'zonemap/n_multiple':    counts['Multiple'],
    }

    if 'recognition' in accumulated:
        rec = accumulated['recognition']
        gt_c, dt_c, edits = rec['gt_chars'], rec['dt_chars'], rec['char_edits']
        correct_c = max(0, min(gt_c, dt_c) - edits)
        metrics['zonemap/char_precision'] = round(correct_c / dt_c, 4) if dt_c > 0 else 0.0
        metrics['zonemap/char_recall']    = round(correct_c / gt_c, 4) if gt_c > 0 else 0.0
        metrics['zonemap/word_precision'] = (
            round(rec['correct_words'] / rec['dt_words'], 4) if rec['dt_words'] > 0 else 0.0
        )
        metrics['zonemap/word_recall'] = (
            round(rec['correct_words'] / rec['gt_words'], 4) if rec['gt_words'] > 0 else 0.0
        )

        cp, cr = metrics['zonemap/char_precision'], metrics['zonemap/char_recall']
        metrics['zonemap/char_f1'] = (
            round(2 * cp * cr / (cp + cr), 4) if (cp + cr) > 0 else 0.0
        )
        wp, wr = metrics['zonemap/word_precision'], metrics['zonemap/word_recall']
        metrics['zonemap/word_f1'] = (
            round(2 * wp * wr / (wp + wr), 4) if (wp + wr) > 0 else 0.0
        )

    return metrics
