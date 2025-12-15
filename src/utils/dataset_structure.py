import glob
from pathlib import Path
from typing import List, Dict, Tuple


def discover_dataset_structure(data_path: str, image_extensions: List[str] = None) -> Dict:
    """
    Analyse la structure du dataset et détermine s'il est plat ou hiérarchique.
    
    Args:
        data_path: Chemin vers le dataset
        image_extensions: Liste des extensions d'images à chercher
        
    Returns:
        Dict contenant:
        - 'type': 'flat' ou 'hierarchical'
        - 'subdirs': Liste des sous-dossiers (si hiérarchique)
        - 'images': Liste de tous les chemins d'images
        - 'structure': Dict mappant sous-dossier -> liste d'images
    """
    if image_extensions is None:
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
    
    data_path = Path(data_path)
    
    # Chercher les images dans le dossier racine
    root_images = []
    for ext in image_extensions:
        root_images.extend(glob.glob(str(data_path / ext)))
    
    # Chercher les sous-dossiers contenant des images
    subdirs_with_images = []
    structure = {}
    
    for item in data_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            subdir_images = []
            for ext in image_extensions:
                subdir_images.extend(glob.glob(str(item / ext)))
            
            if subdir_images:
                subdirs_with_images.append(item)
                structure[str(item)] = sorted(subdir_images)
    
    # Déterminer le type de structure
    if root_images and not subdirs_with_images:
        # Structure plate : images dans le dossier racine
        return {
            'type': 'flat',
            'subdirs': [],
            'images': sorted(root_images),
            'structure': {str(data_path): sorted(root_images)}
        }
    
    elif subdirs_with_images and not root_images:
        # Structure hiérarchique : images dans des sous-dossiers
        all_images = []
        for images in structure.values():
            all_images.extend(images)
        
        return {
            'type': 'hierarchical',
            'subdirs': sorted([str(d) for d in subdirs_with_images]),
            'images': sorted(all_images),
            'structure': structure
        }
    
    elif subdirs_with_images and root_images:
        # Structure mixte : images à la racine ET dans des sous-dossiers
        all_images = sorted(root_images)
        for images in structure.values():
            all_images.extend(images)
        
        structure[str(data_path)] = sorted(root_images)
        
        return {
            'type': 'mixed',
            'subdirs': sorted([str(d) for d in subdirs_with_images]),
            'images': sorted(all_images),
            'structure': structure
        }
    
    else:
        # Aucune image trouvée
        return {
            'type': 'empty',
            'subdirs': [],
            'images': [],
            'structure': {}
        }


def get_output_path_for_image(image_path: str, data_path: str, output_dir: str, 
                               preserve_structure: bool = True) -> Path:
    """
    Détermine le chemin de sortie pour un fichier en préservant la structure.
    
    Args:
        image_path: Chemin de l'image source
        data_path: Racine du dataset d'entrée
        output_dir: Racine du dossier de sortie
        preserve_structure: Si True, préserve les sous-dossiers
        
    Returns:
        Chemin complet du fichier de sortie
    """
    image_path = Path(image_path)
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    
    if not preserve_structure:
        # Structure plate : tout dans output_dir
        return output_dir / image_path.name
    
    # Préserver la structure : calculer le chemin relatif
    try:
        relative_path = image_path.relative_to(data_path)
        output_path = output_dir / relative_path.parent
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / image_path.name
    except ValueError:
        # Si relative_to échoue, utiliser juste le nom
        return output_dir / image_path.name


def process_dataset_hierarchically(data_path: str, output_dir: str, 
                                   process_func, image_extensions: List[str] = None,
                                   preserve_structure: bool = True, **kwargs):
    """
    Traite un dataset en gérant automatiquement la structure plate ou hiérarchique.
    
    Args:
        data_path: Chemin vers le dataset
        output_dir: Dossier de sortie
        process_func: Fonction à appeler pour chaque image/groupe
                     Signature: func(images: List[str], output_path: str, **kwargs)
        image_extensions: Extensions d'images à traiter
        preserve_structure: Si True, préserve l'arborescence en sortie
        **kwargs: Arguments supplémentaires pour process_func
        
    Returns:
        Résultats du traitement
    """
    # Découvrir la structure
    structure_info = discover_dataset_structure(data_path, image_extensions)
    
    if structure_info['type'] == 'empty':
        raise ValueError(f"No images found in {data_path}")
    
    print(f"Dataset structure: {structure_info['type']}")
    print(f"Total images: {len(structure_info['images'])}")
    
    if structure_info['type'] == 'flat':
        # Structure plate : traiter normalement
        print(f"Processing flat structure...")
        return process_func(
            images=structure_info['images'],
            output_dir=output_dir,
            **kwargs
        )
    
    else:
        # Structure hiérarchique : traiter par sous-dossier
        print(f"Processing {len(structure_info['subdirs'])} subdirectories...")
        
        results = []
        for subdir_path, images in structure_info['structure'].items():
            subdir_name = Path(subdir_path).name
            print(f"\n  Processing: {subdir_name} ({len(images)} images)")
            
            # Déterminer le dossier de sortie
            if preserve_structure:
                subdir_output = Path(output_dir) / subdir_name
            else:
                subdir_output = Path(output_dir)
            
            subdir_output.mkdir(parents=True, exist_ok=True)
            
            # Traiter ce sous-dossier
            result = process_func(
                images=images,
                output_dir=str(subdir_output),
                subdir_name=subdir_name,
                **kwargs
            )
            
            results.append({
                'subdir': subdir_name,
                'result': result
            })
        
        return results


def find_corresponding_xml(image_path: str, xml_dir: str = None) -> str:
    """
    Trouve le fichier XML correspondant à une image.
    
    Args:
        image_path: Chemin de l'image
        xml_dir: Dossier où chercher le XML (si différent du dossier de l'image)
        
    Returns:
        Chemin du XML ou None
    """
    image_path = Path(image_path)
    
    # Essayer dans le même dossier que l'image
    xml_path = image_path.with_suffix('.xml')
    if xml_path.exists():
        return str(xml_path)
    
    # Essayer dans xml_dir si spécifié
    if xml_dir:
        xml_path = Path(xml_dir) / image_path.parent.name / image_path.with_suffix('.xml').name
        if xml_path.exists():
            return str(xml_path)
        
        # Essayer directement dans xml_dir (structure plate)
        xml_path = Path(xml_dir) / image_path.with_suffix('.xml').name
        if xml_path.exists():
            return str(xml_path)
    
    return None


def validate_dataset_structure(data_path: str) -> Tuple[bool, str]:
    """
    Valide qu'un dataset est prêt pour DocWorkflow.
    
    Args:
        data_path: Chemin vers le dataset
        
    Returns:
        (is_valid, message)
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        return False, f"Directory does not exist: {data_path}"
    
    if not data_path.is_dir():
        return False, f"Not a directory: {data_path}"
    
    structure_info = discover_dataset_structure(str(data_path))
    
    if structure_info['type'] == 'empty':
        return False, "No images found in dataset"
    
    # Vérifier qu'il y a au moins une image
    if len(structure_info['images']) == 0:
        return False, "Dataset structure detected but no images found"
    
    return True, f"Valid {structure_info['type']} structure with {len(structure_info['images'])} images"


# Export des fonctions principales
__all__ = [
    'discover_dataset_structure',
    'get_output_path_for_image',
    'process_dataset_hierarchically',
    'find_corresponding_xml',
    'validate_dataset_structure'
]