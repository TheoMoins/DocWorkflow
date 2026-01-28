import os
import glob
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import yaml

from src.alto.alto_lines import extract_lines_from_alto


# Mapping des types de lignes vers des class_id
LINE_TYPE_MAPPING = {
    'CustomLine': 0,
    'DefaultLine': 1,
    'DropCapitalLine': 2,
    'HeadingLine': 3,
    'InterlinearLine': 4,
    'MusicLine': 5,
}

# Type par défaut pour les lignes non reconnues
DEFAULT_LINE_TYPE = 'DefaultLine'


def get_line_type_from_tags(tags: Dict[str, str]) -> str:
    """
    Extrait le type de ligne depuis les tags ALTO.
    
    Args:
        tags: Dictionnaire des tags de la ligne
        
    Returns:
        Type de ligne (CustomLine, DefaultLine, etc.)
    """
    if not tags:
        return DEFAULT_LINE_TYPE
    
    # Chercher un tag qui correspond à un type de ligne connu
    for tag_value in tags.values():
        # Le tag peut contenir "line type XXX" ou directement le nom
        tag_lower = tag_value.lower()
        
        for line_type in LINE_TYPE_MAPPING.keys():
            if line_type.lower() in tag_lower:
                return line_type
    
    return DEFAULT_LINE_TYPE


def detect_split_structure(input_dir: str) -> bool:
    """
    Détecte si le répertoire contient une structure train/val/test.
    
    Args:
        input_dir: Répertoire à analyser
        
    Returns:
        True si structure détectée, False sinon
    """
    input_path = Path(input_dir)
    
    # Noms possibles pour chaque split
    train_names = ['train', 'training']
    val_names = ['val', 'valid', 'validation']
    test_names = ['test', 'testing']
    
    # Chercher train
    has_train = False
    for name in train_names:
        split_dir = input_path / name
        if split_dir.exists() and split_dir.is_dir():
            xml_files = list(split_dir.glob('*.xml'))
            if xml_files:
                has_train = True
                break
    
    if not has_train:
        return False
    
    # Chercher au moins val ou test
    has_val_or_test = False
    for name in val_names + test_names:
        split_dir = input_path / name
        if split_dir.exists() and split_dir.is_dir():
            xml_files = list(split_dir.glob('*.xml'))
            if xml_files:
                has_val_or_test = True
                break
    
    return has_val_or_test


def load_existing_split(input_dir: str) -> Dict[str, List[str]]:
    """
    Charge les fichiers depuis une structure train/val/test existante.
    
    Args:
        input_dir: Répertoire contenant les sous-dossiers
        
    Returns:
        Dictionnaire {split_name: [liste de fichiers XML]}
    """
    input_path = Path(input_dir)
    splits = {}
    
    # Mapping des noms possibles vers noms standardisés
    split_mappings = {
        'train': ['train', 'training'],
        'val': ['val', 'valid', 'validation'],
        'test': ['test', 'testing']
    }
    
    for standard_name, possible_names in split_mappings.items():
        for name in possible_names:
            split_dir = input_path / name
            if split_dir.exists() and split_dir.is_dir():
                xml_files = sorted(split_dir.glob('*.xml'))
                if xml_files:
                    splits[standard_name] = [str(f) for f in xml_files]
                    print(f"  Found {standard_name:5} split: {name}/ ({len(xml_files)} files)")
                    break
    
    if 'train' not in splits:
        raise ValueError("Train split not found in input directory")
    
    if 'val' not in splits and 'test' not in splits:
        print("  Warning: No val or test split found")
    
    return splits


def line_to_yolo_format(line: Dict, image_width: int, image_height: int, 
                        line_types_map: Dict[str, int], use_polygon: bool = False) -> Optional[str]:
    """
    Convertit une ligne ALTO en format YOLO (détection ou segmentation).
    
    Args:
        line: Dictionnaire de ligne avec boundary/baseline et tags
        image_width: Largeur de l'image
        image_height: Hauteur de l'image
        line_types_map: Mapping type -> class_id
        use_polygon: Si True, génère format segmentation (polygone complet)
                     Si False, génère format détection (bounding box)
        
    Returns:
        Ligne YOLO formatée ou None si conversion impossible
    """
    # Extraire le type de ligne
    line_type = get_line_type_from_tags(line.get('tags', {}))
    class_id = line_types_map.get(line_type, line_types_map[DEFAULT_LINE_TYPE])
    
    # Utiliser boundary si disponible, sinon créer depuis baseline
    if line.get('boundary') and line['boundary']:
        boundary = line['boundary']
    elif line.get('baseline') and len(line['baseline']) >= 2:
        # Créer une boundary avec buffer de 20px autour de la baseline
        baseline = line['baseline']
        buffer = 20
        
        xs = [p[0] for p in baseline]
        ys = [p[1] for p in baseline]
        
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys) - buffer
        max_y = max(ys) + buffer
        
        boundary = [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ]
    else:
        return None
    
    if use_polygon:
        # Format YOLO segmentation : class_id x1 y1 x2 y2 x3 y3 ... xn yn
        # Tous les points du polygone normalisés
        normalized_points = []
        for point in boundary:
            x_norm = max(0, min(1, point[0] / image_width))
            y_norm = max(0, min(1, point[1] / image_height))
            normalized_points.append(f"{x_norm:.6f} {y_norm:.6f}")
        
        return f"{class_id} {' '.join(normalized_points)}"
    
    else:
        # Format YOLO détection : class_id center_x center_y width height
        # Calculer bounding box depuis boundary
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        
        # Convertir en format YOLO (centre x, centre y, largeur, hauteur, normalisés)
        center_x = (min_x + max_x) / 2 / image_width
        center_y = (min_y + max_y) / 2 / image_height
        width = (max_x - min_x) / image_width
        height = (max_y - min_y) / image_height
        
        # Limiter aux valeurs valides [0, 1]
        center_x = max(0, min(1, center_x))
        center_y = max(0, min(1, center_y))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        return f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"


def find_image_for_xml(xml_path: str) -> Optional[str]:
    """
    Trouve l'image correspondant à un fichier XML ALTO.
    
    Args:
        xml_path: Chemin du fichier XML
        
    Returns:
        Chemin de l'image ou None
    """
    # Extraire info depuis ALTO
    try:
        image_path, _, _ = extract_lines_from_alto(xml_path)
        if image_path and os.path.exists(image_path):
            return image_path
    except Exception:
        pass
    
    # Chercher dans le même dossier que le XML
    xml_dir = os.path.dirname(xml_path)
    base_name = os.path.splitext(os.path.basename(xml_path))[0]
    
    for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
        img_path = os.path.join(xml_dir, base_name + ext)
        if os.path.exists(img_path):
            return img_path
    
    # Chercher dans le dossier parent (cas où images et XML séparés)
    parent_dir = os.path.dirname(xml_dir)
    for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
        img_path = os.path.join(parent_dir, base_name + ext)
        if os.path.exists(img_path):
            return img_path
    
    return None


def process_split(xml_files: List[str], output_dir: Path, split_name: str,
                  line_types_map: Dict[str, int], use_polygon: bool = False) -> Tuple[int, int]:
    """
    Traite un split (train/val/test) et génère les fichiers YOLO.
    
    Args:
        xml_files: Liste des fichiers XML à traiter
        output_dir: Répertoire de sortie YOLO
        split_name: Nom du split (train/val/test)
        line_types_map: Mapping type -> class_id
        use_polygon: Si True, génère format segmentation (polygones)
        
    Returns:
        Tuple (nb_images_traitées, nb_lignes_totales)
    """
    images_dir = output_dir / 'images' / split_name
    labels_dir = output_dir / 'labels' / split_name
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    processed_images = 0
    total_lines = 0
    
    for xml_path in xml_files:
        try:
            # Extraire lignes
            _, lines, _ = extract_lines_from_alto(xml_path)
            
            if not lines:
                continue
            
            # Trouver image
            image_path = find_image_for_xml(xml_path)
            if not image_path:
                print(f"  Warning: No image found for {os.path.basename(xml_path)}")
                continue
            
            # Obtenir dimensions image
            with Image.open(image_path) as img:
                img_width, img_height = img.size
            
            # Convertir lignes en format YOLO
            yolo_lines = []
            for line in lines:
                yolo_line = line_to_yolo_format(line, img_width, img_height, 
                                               line_types_map, use_polygon)
                if yolo_line:
                    yolo_lines.append(yolo_line)
            
            if not yolo_lines:
                continue
            
            # Copier image
            base_name = os.path.splitext(os.path.basename(xml_path))[0]
            img_ext = os.path.splitext(image_path)[1]
            
            dest_image = images_dir / f"{base_name}{img_ext}"
            shutil.copy2(image_path, dest_image)
            
            # Écrire labels
            dest_label = labels_dir / f"{base_name}.txt"
            with open(dest_label, 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            processed_images += 1
            total_lines += len(yolo_lines)
            
        except Exception as e:
            print(f"  Error processing {os.path.basename(xml_path)}: {e}")
            continue
    
    return processed_images, total_lines


def create_yolo_config(output_dir: Path, splits: Dict[str, int],
                       line_types_map: Dict[str, int]) -> None:
    """
    Crée le fichier data.yaml pour YOLO.
    
    Args:
        output_dir: Répertoire racine du dataset YOLO
        splits: Dictionnaire des splits avec leur nombre d'images
        line_types_map: Mapping type -> class_id
    """
    config = {
        'path': str(output_dir.absolute()),
        'train': 'images/train' if 'train' in splits else None,
        'val': 'images/val' if 'val' in splits else None,
        'test': 'images/test' if 'test' in splits else None,
        'names': {v: k for k, v in line_types_map.items()}
    }
    
    # Supprimer les splits non utilisés
    config = {k: v for k, v in config.items() if v is not None}
    
    config_path = output_dir / 'data.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✓ Configuration saved: {config_path}")


def convert_alto_lines_to_yolo(input_dir: str, output_dir: str, use_polygon: bool = False) -> None:
    """
    Convertit un dataset ALTO (lignes) vers format YOLO.
    Préserve la structure train/val/test existante.
    
    Args:
        input_dir: Répertoire contenant les fichiers ALTO avec structure train/val/test
        output_dir: Répertoire de sortie pour le dataset YOLO
        use_polygon: Si True, génère format segmentation (polygones)
                     Si False, génère format détection (bounding boxes)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    format_type = "SEGMENTATION (polygons)" if use_polygon else "DETECTION (bounding boxes)"
    print(f"Converting ALTO lines to YOLO format...")
    print(f"Format: {format_type}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # Détection de la structure
    if not detect_split_structure(input_dir):
        raise ValueError(
            "No train/val/test split structure detected in input directory.\n"
            "Expected structure:\n"
            "  input_dir/\n"
            "    train/*.xml + images\n"
            "    val/*.xml + images\n"
            "    test/*.xml + images"
        )
    
    print("✓ Detected existing train/val/test split structure")
    print("  Preserving original split...")
    print()
    
    # Charger les splits
    splits = load_existing_split(input_dir)
    print()
    
    # Créer structure YOLO
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Traiter chaque split
    print("Processing splits...")
    split_stats = {}
    
    for split_name, xml_files in splits.items():
        print(f"\n  {split_name.upper()} split:")
        n_images, n_lines = process_split(xml_files, output_path, split_name, 
                                          LINE_TYPE_MAPPING, use_polygon)
        split_stats[split_name] = n_images
        print(f"    ✓ {n_images} images, {n_lines} lines")
    
    # Créer config YOLO
    create_yolo_config(output_path, split_stats, LINE_TYPE_MAPPING)
    
    # Résumé
    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"Format: {format_type}")
    print(f"Output directory: {output_dir}")
    print(f"Total images: {sum(split_stats.values())}")
    print(f"Line types: {len(LINE_TYPE_MAPPING)}")
    print()
    print("Line type mapping:")
    for line_type, class_id in sorted(LINE_TYPE_MAPPING.items(), key=lambda x: x[1]):
        print(f"  {class_id}: {line_type}")
    
    if use_polygon:
        print(f"\nTraining command (SEGMENTATION):")
        print(f"  yolo segment train data={output_dir}/data.yaml model=yolo11n-seg.pt epochs=100")
    else:
        print(f"\nTraining command (DETECTION):")
        print(f"  yolo detect train data={output_dir}/data.yaml model=yolo11n.pt epochs=100")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert ALTO line annotations to YOLO format"
    )
    parser.add_argument("input_dir", help="Input directory with train/val/test structure")
    parser.add_argument("output_dir", help="Output directory for YOLO dataset")
    parser.add_argument(
        "--polygon", 
        action="store_true",
        help="Generate YOLO segmentation format (polygons) instead of detection format (bboxes)"
    )
    
    args = parser.parse_args()
    
    convert_alto_lines_to_yolo(args.input_dir, args.output_dir, args.polygon)