"""
Convert ALTO TextLines to YOLO format for training line segmentation models.
"""

import os
import shutil
import yaml
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple, Optional
import random

from src.alto.alto_lines import extract_lines_from_alto


def detect_split_structure(alto_dir: Path) -> bool:
    """
    Detect if directory has existing train/val/test split structure.
    
    Args:
        alto_dir: Path to ALTO directory
    
    Returns:
        True if split structure exists, False otherwise
    """
    # Check for common split directory names
    possible_splits = {
        'train': ['train', 'training'],
        'val': ['val', 'valid', 'validation'],
        'test': ['test', 'testing']
    }
    
    found_splits = set()
    
    for split_type, possible_names in possible_splits.items():
        for name in possible_names:
            split_path = alto_dir / name
            if split_path.exists() and split_path.is_dir():
                # Check if it contains ALTO files
                xml_files = list(split_path.glob("*.xml"))
                if xml_files:
                    found_splits.add(split_type)
                    break
    
    # Consider it a split structure if we have at least train + one other
    return 'train' in found_splits and len(found_splits) >= 2


def load_existing_split(alto_dir: Path) -> dict:
    """
    Load files from existing train/val/test split structure.
    
    Args:
        alto_dir: Path to ALTO directory with subdirectories
    
    Returns:
        Dictionary mapping split names to list of file paths
    """
    # Map common directory names to standard split names
    split_mapping = {
        'train': ['train', 'training'],
        'val': ['val', 'valid', 'validation'],
        'test': ['test', 'testing']
    }
    
    split_files = {
        'train': [],
        'val': [],
        'test': []
    }
    
    for split_name, possible_dirs in split_mapping.items():
        for dir_name in possible_dirs:
            split_path = alto_dir / dir_name
            if split_path.exists() and split_path.is_dir():
                xml_files = sorted(list(split_path.glob("*.xml")))
                if xml_files:
                    split_files[split_name] = xml_files
                    print(f"  Found {split_name:5s} split: {dir_name}/ ({len(xml_files)} files)")
                    break
    
    # Validate that we found all splits
    missing_splits = [split for split, files in split_files.items() if not files]
    if missing_splits:
        print(f"\n⚠ Warning: Missing splits: {', '.join(missing_splits)}")
        print(f"  These splits will be empty in the output.")
    
    return split_files


def convert_alto_lines_to_yolo(
    alto_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    class_mapping: Optional[dict] = None,
    preserve_split: bool = True
):
    """
    Convert ALTO XML files with TextLines to YOLO format.
    
    Args:
        alto_dir: Directory containing ALTO XML files and images
                 Can be a flat directory or contain train/val/test subdirectories
        output_dir: Output directory for YOLO dataset
        train_ratio: Ratio of data for training (default: 0.8)
                    Only used if preserve_split=False
        val_ratio: Ratio of data for validation (default: 0.1)
                  Only used if preserve_split=False
        test_ratio: Ratio of data for testing (default: 0.1)
                   Only used if preserve_split=False
        seed: Random seed for reproducible splits
             Only used if preserve_split=False
        class_mapping: Optional dict mapping line tags/types to class IDs
                      If None, all lines are class 0
        preserve_split: If True, detect and preserve existing train/val/test splits
                       If False, perform random split (default: True)
    
    Returns:
        Dict with conversion statistics
    """
    
    output_dir = Path(output_dir)
    alto_dir = Path(alto_dir)
    
    print(f"\n{'='*60}")
    print(f"ALTO LINES → YOLO CONVERSION")
    print(f"{'='*60}\n")
    print(f"Source: {alto_dir}")
    print(f"Output: {output_dir}")
    
    # Detect if input has existing split structure
    has_existing_split = detect_split_structure(alto_dir)
    
    if has_existing_split and preserve_split:
        print(f"✓ Detected existing train/val/test split structure")
        print(f"  Preserving original split...")
        split_files = load_existing_split(alto_dir)
    else:
        if has_existing_split:
            print(f"⚠ Existing split detected but preserve_split=False")
            print(f"  Performing random split...")
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
            raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
        
        print(f"Split: train={train_ratio:.0%}, val={val_ratio:.0%}, test={test_ratio:.0%}")
        
        # Find all ALTO files
        alto_files = sorted(list(alto_dir.glob("*.xml")))
        
        if not alto_files:
            raise ValueError(f"No ALTO XML files found in {alto_dir}")
        
        print(f"Found {len(alto_files)} ALTO files\n")
        
        # Split files into train/val/test
        random.seed(seed)
        random.shuffle(alto_files)
        
        n_train = int(len(alto_files) * train_ratio)
        n_val = int(len(alto_files) * val_ratio)
        
        split_files = {
            'train': alto_files[:n_train],
            'val': alto_files[n_train:n_train + n_val],
            'test': alto_files[n_train + n_val:]
        }
    
    # Create output directory structure
    splits = ['train', 'val', 'test']
    for split in splits:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    print(f"\nSplit distribution:")
    for split, files in split_files.items():
        print(f"  {split:5s}: {len(files):4d} files")
    print()
    
    # Process files
    stats = {
        'total_files': 0,
        'total_lines': 0,
        'skipped_files': 0,
        'class_distribution': {},
        'splits': {}
    }
    
    for split, files in split_files.items():
        split_stats = process_split(
            files=files,
            split=split,
            output_dir=output_dir,
            class_mapping=class_mapping,
            source_is_split=has_existing_split and preserve_split
        )
        stats['splits'][split] = split_stats
        stats['total_files'] += split_stats['processed_files']
        stats['total_lines'] += split_stats['total_lines']
        stats['skipped_files'] += split_stats['skipped_files']
        
        # Merge class distributions
        for cls, count in split_stats['class_distribution'].items():
            stats['class_distribution'][cls] = stats['class_distribution'].get(cls, 0) + count
    
    # Create data.yaml
    create_data_yaml(
        output_dir=output_dir,
        class_mapping=class_mapping,
        class_distribution=stats['class_distribution']
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"✓ Processed: {stats['total_files']} files")
    print(f"✓ Total lines: {stats['total_lines']}")
    if stats['skipped_files'] > 0:
        print(f"⚠ Skipped: {stats['skipped_files']} files (no lines or image not found)")
    
    print(f"\nClass distribution:")
    for cls, count in sorted(stats['class_distribution'].items()):
        print(f"  Class {cls}: {count:6d} lines")
    
    print(f"\nDataset ready at: {output_dir}")
    print(f"Use this path in your training config:")
    print(f"  data: \"{output_dir / 'data.yaml'}\"")
    print(f"{'='*60}\n")
    
    return stats


def process_split(
    files: List[Path],
    split: str,
    output_dir: Path,
    class_mapping: Optional[dict] = None,
    source_is_split: bool = False
) -> dict:
    """
    Process a single split (train/val/test).
    
    Args:
        files: List of ALTO file paths
        split: Split name ('train', 'val', or 'test')
        output_dir: Output directory
        class_mapping: Optional class mapping
        source_is_split: If True, files are already in split subdirectories
    
    Returns:
        Statistics dictionary
    """
    stats = {
        'processed_files': 0,
        'skipped_files': 0,
        'total_lines': 0,
        'class_distribution': {}
    }
    
    images_dir = output_dir / 'images' / split
    labels_dir = output_dir / 'labels' / split
    
    print(f"Processing {split} split...")
    
    for alto_file in tqdm(files, desc=f"  {split}", unit="file"):
        try:
            # Extract lines from ALTO
            image_path, lines, regions = extract_lines_from_alto(str(alto_file))
            
            # Handle case where image_path is relative to ALTO file location
            if not image_path or not os.path.exists(image_path):
                # Try to find image in same directory as ALTO file
                alto_parent = alto_file.parent
                image_name = Path(image_path).name if image_path else None
                
                if not image_name:
                    # Try common extensions
                    base_name = alto_file.stem
                    for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                        potential_path = alto_parent / f"{base_name}{ext}"
                        if potential_path.exists():
                            image_path = str(potential_path)
                            break
                else:
                    potential_path = alto_parent / image_name
                    if potential_path.exists():
                        image_path = str(potential_path)
            
            if not image_path or not os.path.exists(image_path):
                print(f"\n  ⚠ Image not found for {alto_file.name}")
                stats['skipped_files'] += 1
                continue
            
            if not lines:
                stats['skipped_files'] += 1
                continue
            
            # Get image dimensions
            with Image.open(image_path) as img:
                width, height = img.size
            
            # Copy image
            img_name = Path(image_path).name
            shutil.copy2(image_path, images_dir / img_name)
            
            # Convert lines to YOLO format
            yolo_lines = []
            for line in lines:
                yolo_line = convert_line_to_yolo(
                    line=line,
                    image_width=width,
                    image_height=height,
                    class_mapping=class_mapping
                )
                
                if yolo_line:
                    yolo_lines.append(yolo_line)
                    # Update class distribution
                    class_id = int(yolo_line.split()[0])
                    stats['class_distribution'][class_id] = \
                        stats['class_distribution'].get(class_id, 0) + 1
            
            # Save YOLO label file
            if yolo_lines:
                label_file = labels_dir / f"{Path(img_name).stem}.txt"
                with open(label_file, 'w') as f:
                    f.write('\n'.join(yolo_lines))
                
                stats['processed_files'] += 1
                stats['total_lines'] += len(yolo_lines)
            else:
                stats['skipped_files'] += 1
        
        except Exception as e:
            print(f"\n  Error processing {alto_file.name}: {e}")
            stats['skipped_files'] += 1
            continue
    
    return stats


def convert_line_to_yolo(
    line: dict,
    image_width: int,
    image_height: int,
    class_mapping: Optional[dict] = None
) -> Optional[str]:
    """
    Convert a single line to YOLO format.
    
    Args:
        line: Line dictionary with 'boundary' or 'baseline'
        image_width: Image width in pixels
        image_height: Image height in pixels
        class_mapping: Optional mapping of line tags to class IDs
    
    Returns:
        YOLO format string: "class x_center y_center width height"
        or None if line cannot be converted
    """
    
    # Get bounding box from boundary or baseline
    if line.get('boundary'):
        boundary = np.array(line['boundary'])
        x_min, y_min = boundary.min(axis=0)
        x_max, y_max = boundary.max(axis=0)
    
    elif line.get('baseline') and len(line['baseline']) >= 2:
        # Create bounding box from baseline with buffer
        baseline = np.array(line['baseline'])
        x_min, x_max = baseline[:, 0].min(), baseline[:, 0].max()
        y_min, y_max = baseline[:, 1].min(), baseline[:, 1].max()
        
        # Add vertical buffer (typical line height ~30-50px)
        buffer = 20
        y_min = max(0, y_min - buffer)
        y_max = min(image_height, y_max + buffer)
    
    else:
        return None
    
    # Ensure valid box
    if x_max <= x_min or y_max <= y_min:
        return None
    
    # Determine class ID
    if class_mapping and 'tags' in line and line['tags']:
        # Use first tag if available
        tag = line['tags'][0] if isinstance(line['tags'], list) else line['tags']
        class_id = class_mapping.get(tag, 0)
    else:
        class_id = 0  # Default: all lines are class 0
    
    # Normalize to [0, 1]
    x_center = ((x_min + x_max) / 2) / image_width
    y_center = ((y_min + y_max) / 2) / image_height
    box_width = (x_max - x_min) / image_width
    box_height = (y_max - y_min) / image_height
    
    # Clip to valid range
    x_center = np.clip(x_center, 0, 1)
    y_center = np.clip(y_center, 0, 1)
    box_width = np.clip(box_width, 0, 1)
    box_height = np.clip(box_height, 0, 1)
    
    # YOLO format: class x_center y_center width height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def create_data_yaml(
    output_dir: Path,
    class_mapping: Optional[dict] = None,
    class_distribution: Optional[dict] = None
):
    """
    Create data.yaml file for YOLO training.
    
    Args:
        output_dir: Output directory
        class_mapping: Optional class mapping
        class_distribution: Optional class distribution for reference
    """
    
    # Determine number of classes and names
    if class_mapping:
        nc = len(class_mapping)
        # Reverse mapping: class_id -> name
        names = [''] * nc
        for name, class_id in class_mapping.items():
            names[class_id] = name
    else:
        nc = 1
        names = ['line']
    
    data = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': nc,
        'names': names
    }
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        # Add class distribution as comment
        if class_distribution:
            f.write('\n# Class distribution:\n')
            for cls_id, count in sorted(class_distribution.items()):
                cls_name = names[cls_id] if cls_id < len(names) else f'class_{cls_id}'
                f.write(f'#   {cls_name}: {count} lines\n')
    
    print(f"\n✓ Created {yaml_path}")


def discover_line_classes(alto_dir: str) -> dict:
    """
    Discover unique line tags/types in ALTO files.
    
    Args:
        alto_dir: Directory containing ALTO files
    
    Returns:
        Dictionary mapping tag names to suggested class IDs
    """
    alto_dir = Path(alto_dir)
    alto_files = list(alto_dir.glob("*.xml"))
    
    unique_tags = set()
    
    print("Discovering line types...")
    for alto_file in tqdm(alto_files, desc="Scanning", unit="file"):
        try:
            _, lines, _ = extract_lines_from_alto(str(alto_file))
            
            for line in lines:
                if 'tags' in line and line['tags']:
                    tags = line['tags'] if isinstance(line['tags'], list) else [line['tags']]
                    unique_tags.update(tags)
        
        except Exception:
            continue
    
    if not unique_tags:
        print("\nNo line tags found. All lines will be class 0.")
        return {}
    
    # Create mapping
    class_mapping = {tag: idx for idx, tag in enumerate(sorted(unique_tags))}
    
    print(f"\nFound {len(class_mapping)} line types:")
    for tag, class_id in class_mapping.items():
        print(f"  {class_id}: {tag}")
    
    return class_mapping