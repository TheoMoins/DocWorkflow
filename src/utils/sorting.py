import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from typing import List, Optional


def _get_zone_first_line_y(zone_info: dict, lines_with_blocks: Optional[list]) -> float:
    if lines_with_blocks is None:
        return zone_info['bbox'][1]
    if zone_info.get('is_original', False):
        zone_lines = [i for i in lines_with_blocks if i['block'] == zone_info['block']]
        if zone_lines:
            return min(line['y_pos'] for line in zone_lines)
    return zone_info['bbox'][1]


def _infer_image_shape(zones: list) -> tuple:
    """Infère (height, width) depuis les bboxes quand l'image n'est pas disponible."""
    all_x2 = [z['bbox'][2] for z in zones]
    all_y2 = [z['bbox'][3] for z in zones]
    return (int(max(all_y2)) + 1, int(max(all_x2)) + 1)


def _moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    return np.convolve(values, np.ones(window_size), "same") / window_size


def _build_heatmap(zones: list, image_shape: tuple) -> np.ndarray:
    heatmap = np.full(image_shape[:2], fill_value=255, dtype=np.uint8)
    for z in zones:
        x1, y1, x2, y2 = map(int, z['bbox'])
        heatmap[y1:y2, x1:x2] = 0
    return heatmap


def _get_horizontal_breakpoints(heatmap: np.ndarray, window_size: int = 10) -> List[int]:
    horizontal_whites = np.where(
        _moving_average((heatmap != 255).sum(axis=1), window_size).astype(int) == 0
    )[0]
    breakpoints, window = [], []
    for idx in horizontal_whites:
        if not window:
            window.append(idx)
        elif window[-1] != idx - 1:
            breakpoints.append(window[len(window) // 2])
            window = [idx]
        else:
            window.append(idx)
    if window:
        breakpoints.append(window[len(window) // 2])
    return breakpoints


def _get_vertical_breakpoints(heatmap_band: np.ndarray, divisor: int = 4,
                               window_size: int = 100) -> np.ndarray:
    values = (heatmap_band != 255).sum(axis=0)
    moving_avg = _moving_average(values, window_size)
    return find_peaks(-moving_avg, prominence=heatmap_band.shape[0] // divisor)[0]


# ---------------------------------------------------------------------------
# Strategy 1 : no sort
# ---------------------------------------------------------------------------

def sort_none(zones: list, **kwargs) -> list:
    return zones


# ---------------------------------------------------------------------------
# Strategy 2 : DBSCAN (actual method)
# ---------------------------------------------------------------------------

def sort_dbscan(zones: list, lines_with_blocks: Optional[list] = None,
                eps: int = 300, min_samples: int = 1, **kwargs) -> list:
    """
    Clustering DBSCAN sur l'axe X pour détecter les colonnes.
    Tri colonne par colonne, de gauche à droite, puis de haut en bas.
    """
    if len(zones) <= 1:
        return zones

    centers_x = np.array([(z['bbox'][0] + z['bbox'][2]) / 2 for z in zones]).reshape(-1, 1)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(centers_x).labels_

    columns = {}
    for idx, label in enumerate(labels):
        z = zones[idx]
        first_y = _get_zone_first_line_y(z, lines_with_blocks)
        center_x = (z['bbox'][0] + z['bbox'][2]) / 2
        columns.setdefault(label, []).append({'zone': z, 'first_line_y': first_y, 'center_x': center_x})

    sorted_columns = sorted(
        [(sum(e['center_x'] for e in col) / len(col), col) for col in columns.values()],
        key=lambda t: t[0]
    )

    result = []
    for _, col in sorted_columns:
        col.sort(key=lambda z: (z['first_line_y'], z['center_x']))
        result.extend(z['zone'] for z in col)
    return result


# ---------------------------------------------------------------------------
# Stratégie 3 : Heatmap (from Chahan Script)
# ---------------------------------------------------------------------------

def sort_heatmap(zones: list, lines_with_blocks: Optional[list] = None,
                 image_shape: Optional[tuple] = None,
                 median_multiplier: int = 5, window_size: int = 15,
                 divisor: int = 4, **kwargs) -> list:
    """
    Tri basé sur une heatmap binaire.
    1. Détection de bandes blanches horizontales → blocs verticaux
    2. Pour les blocs hauts : détection de gouttières verticales → colonnes
    3. Tri par bloc, puis par colonne, puis par Y dans chaque colonne
    """
    if len(zones) <= 1:
        return zones

    if image_shape is None:
        image_shape = _infer_image_shape(zones)

    heatmap = _build_heatmap(zones, image_shape)

    # Calcul de la hauteur médiane des zones
    heights = [z['bbox'][3] - z['bbox'][1] for z in zones]
    height_median = float(np.median(heights)) if heights else 1.0

    h_breakpoints = [0] + _get_horizontal_breakpoints(
        heatmap, window_size=max(1, int(height_median / 2))
    ) + [image_shape[0]]

    # Assignation de chaque zone à un (band_id, column_id) → tri final
    zone_order = []
    for band_id, (y_min, y_max) in enumerate(zip(h_breakpoints[:-1], h_breakpoints[1:])):
        band_zones = [z for z in zones
                      if y_min < (z['bbox'][1] + z['bbox'][3]) / 2 <= y_max]
        if not band_zones:
            continue

        if y_max - y_min < median_multiplier * height_median:
            # Bande fine : une seule colonne implicite
            for z in sorted(band_zones, key=lambda z: (
                _get_zone_first_line_y(z, lines_with_blocks), z['bbox'][0]
            )):
                zone_order.append((band_id, 0, _get_zone_first_line_y(z, lines_with_blocks), z['bbox'][0], z))
        else:
            # Bande large : chercher des colonnes par creux verticaux
            v_breakpoints = [0] + list(
                _get_vertical_breakpoints(heatmap[y_min:y_max], divisor, window_size)
            ) + [image_shape[1] + 1]
            for col_id, (x_min, x_max) in enumerate(zip(v_breakpoints[:-1], v_breakpoints[1:])):
                col_zones = [z for z in band_zones
                             if x_min < (z['bbox'][0] + z['bbox'][2]) / 2 <= x_max]
                for z in col_zones:
                    zone_order.append((band_id, col_id,
                                       _get_zone_first_line_y(z, lines_with_blocks),
                                       z['bbox'][0], z))

    zone_order.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
    return [t[4] for t in zone_order]


# ---------------------------------------------------------------------------
# Stratégie 4 : Hybrid (segmentation horizontale heatmap + DBSCAN)
# ---------------------------------------------------------------------------

def sort_hybrid(zones: list, lines_with_blocks: Optional[list] = None,
                image_shape: Optional[tuple] = None,
                eps: int = 300, min_samples: int = 1,
                median_multiplier: int = 5, window_size: int = 15, **kwargs) -> list:
    """
    Hybride : découpe d'abord les bandes horizontales par heatmap,
    puis applique DBSCAN pour détecter les colonnes dans chaque bande.
    Combine la robustesse de la segmentation verticale avec la souplesse
    du clustering pour les colonnes irrégulières.
    """
    if len(zones) <= 1:
        return zones

    if image_shape is None:
        image_shape = _infer_image_shape(zones)

    heatmap = _build_heatmap(zones, image_shape)
    heights = [z['bbox'][3] - z['bbox'][1] for z in zones]
    height_median = float(np.median(heights)) if heights else 1.0

    h_breakpoints = [0] + _get_horizontal_breakpoints(
        heatmap, window_size=max(1, int(height_median / 2))
    ) + [image_shape[0]]

    result = []
    for y_min, y_max in zip(h_breakpoints[:-1], h_breakpoints[1:]):
        band_zones = [z for z in zones
                      if y_min < (z['bbox'][1] + z['bbox'][3]) / 2 <= y_max]
        if not band_zones:
            continue
        # DBSCAN sur la bande courante avec eps adapté à sa largeur
        band_eps = eps if y_max - y_min >= median_multiplier * height_median else image_shape[1]
        sorted_band = sort_dbscan(band_zones, lines_with_blocks,
                                  eps=band_eps, min_samples=min_samples)
        result.extend(sorted_band)

    return result


SORTING_STRATEGIES = {
    "none": sort_none,
    "dbscan": sort_dbscan,
    "heatmap": sort_heatmap,
    "hybrid": sort_hybrid,
}


def get_sorting_strategy(method: str):
    if method not in SORTING_STRATEGIES:
        raise ValueError(f"Méthode de tri inconnue : '{method}'. "
                         f"Valeurs acceptées : {list(SORTING_STRATEGIES)}")
    return SORTING_STRATEGIES[method]

def sort_zones_reading_order(zones, lines_with_blocks=None, eps=300, min_samples=1,
                              method="dbscan", image_shape=None):
    """
    Trier les zones en ordre de lecture.

    Args:
        zones: Liste de zones (dict avec clé 'bbox': [x1, y1, x2, y2])
        lines_with_blocks: Lignes assignées aux zones (optionnel)
        eps: Distance DBSCAN (utilisé par les méthodes dbscan et hybrid)
        min_samples: Minimum de zones par cluster DBSCAN
        method: Stratégie de tri — 'none' | 'dbscan' | 'heatmap' | 'hybrid'
        image_shape: (height, width) de l'image source (inféré si None)

    Returns:
        Liste triée des zones
    """
    from src.utils.sorting import get_sorting_strategy
    if not zones:
        return zones
    strategy = get_sorting_strategy(method)
    return strategy(zones, lines_with_blocks=lines_with_blocks, eps=eps,
                    min_samples=min_samples, image_shape=image_shape)