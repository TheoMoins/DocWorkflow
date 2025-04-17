import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from lxml import etree as ET
import numpy as np
from collections import defaultdict
import os
import glob
from pathlib import Path
import statistics
from typing import List, Dict, Tuple, Optional, Any

class DocumentVisualizer:
    """
    Classe pour visualiser et analyser les résultats de segmentation de documents
    à partir de fichiers ALTO XML
    """
    
    def __init__(self, image_path, alto_path, visualization_type='layout'):
        """
        Initialise le visualiseur de document.
        
        Args:
            image_path: Chemin vers l'image du document
            alto_path: Chemin vers le fichier ALTO XML correspondant
            visualization_type: Type de visualisation ('layout' ou 'line')
        """
        self.image_path = image_path
        self.alto_path = alto_path
        self.visualization_type = visualization_type
        self.image = None
        self.root = None
        self.ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
        self.tag_labels = {}
        
        self._load_data()
    
    def _load_data(self):
        """Charge l'image et le fichier ALTO XML"""
        try:
            self.image = Image.open(self.image_path)
        except Exception as e:
            print(f"Error loading image {self.image_path}: {e}")
            return False
            
        try:
            tree = ET.parse(self.alto_path)
            self.root = tree.getroot()
            self.tag_labels = self._load_tag_labels()
            return True
        except Exception as e:
            print(f"Error loading ALTO file {self.alto_path}: {e}")
            return False
    
    def _load_tag_labels(self):
        """Charge les correspondances entre TAGREFs et labels depuis la section Tags"""
        tag_labels = {}
        tags_section = self.root.find('.//alto:Tags', self.ns)
        
        if tags_section is not None:
            for tag in tags_section.findall('.//alto:OtherTag', self.ns):
                tag_id = tag.get('ID')
                label = tag.get('LABEL')
                if tag_id and label:
                    tag_labels[tag_id] = label
                    
        return tag_labels
    
    @staticmethod
    def safe_int(value, default=0):
        """Convertit une valeur en entier de manière sécurisée"""
        try:
            return int(float(value)) if value is not None else default
        except (ValueError, TypeError):
            return default
            
    @staticmethod
    def safe_float(value, default=0.5):
        """Convertit une valeur en flottant de manière sécurisée"""
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    def get_bbox_from_shape(self, element):
        """
        Extrait les coordonnées de la boîte englobante à partir d'un élément Shape.
        
        Args:
            element: Élément XML contenant un Shape avec un Polygon
            
        Returns:
            Liste [x, y, width, height] ou None
        """
        shape = element.find('.//alto:Shape', self.ns)
        if shape is not None:
            polygon = shape.find('.//alto:Polygon', self.ns)
            if polygon is not None:
                points = polygon.get('POINTS', '').split()
                if points:
                    coords = []
                    for p in points:
                        parts = p.split(',')
                        if len(parts) == 2:
                            try:
                                coords.append((int(parts[0]), int(parts[1])))
                            except ValueError:
                                continue
                            
                    if coords:
                        x_coords, y_coords = zip(*coords)
                        return [min(x_coords), min(y_coords), 
                               max(x_coords) - min(x_coords), 
                               max(y_coords) - min(y_coords)]
        return None
    
    def get_bbox(self, element):
        """
        Obtient la boîte englobante d'un élément, soit à partir de ses attributs,
        soit à partir de sa forme (Shape).
        
        Args:
            element: Élément XML
            
        Returns:
            Liste [x, y, width, height]
        """
        bbox = [
            self.safe_int(element.get('HPOS')),
            self.safe_int(element.get('VPOS')),
            self.safe_int(element.get('WIDTH')),
            self.safe_int(element.get('HEIGHT'))
        ]
        
        if all(v == 0 for v in bbox):
            shape_bbox = self.get_bbox_from_shape(element)
            if shape_bbox:
                return shape_bbox
        return bbox
    
    def analyze_layout(self):
        """
        Analyse la segmentation en blocs.
        
        Returns:
            Liste de dictionnaires contenant les informations de chaque bloc
        """
        blocks = []
        
        for block in self.root.findall('.//alto:TextBlock', self.ns):
            bbox = self.get_bbox(block)
            tag_ref = block.get('TAGREFS', 'unknown')
            block_label = self.tag_labels.get(tag_ref, "unknown")
            
            blocks.append({
                'bbox': bbox,
                'type': tag_ref,
                'label': block_label,
                'n_lines': len(block.findall('.//alto:TextLine', self.ns))
            })
            
        return blocks
    
    def analyze_lines(self):
        """
        Analyse la segmentation en lignes.
        
        Returns:
            Liste de dictionnaires contenant les informations de chaque ligne
        """
        lines = []
        
        for line in self.root.findall('.//alto:TextLine', self.ns):
            bbox = self.get_bbox(line)
            baseline = line.get('BASELINE', '')
            
            # Trouver le bloc parent
            parent_block = line.getparent()
            block_tag_ref = parent_block.get('TAGREFS', 'unknown') if parent_block is not None else 'unknown'
            block_label = self.tag_labels.get(block_tag_ref, "unknown") if parent_block is not None else "unknown"
            
            lines.append({
                'bbox': bbox,
                'baseline': baseline,
                'block_type': block_label
            })
            
        return lines
    
    def draw_layout(self, figsize=(15, 20), save_path=None):
        """
        Visualise la segmentation en blocs.
        
        Args:
            figsize: Taille de la figure (width, height)
            save_path: Chemin pour sauvegarder l'image (si None, affiche la figure)
            
        Returns:
            Figure matplotlib
        """
        blocks = self.analyze_layout()
        
        # Séparer les blocs MainZone des autres
        main_blocks = [b for b in blocks if b['label'] == 'MainZone']
        other_blocks = [b for b in blocks if b['label'] != 'MainZone']
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.image)
        
        # Générer une palette de couleurs pour les MainZone
        main_colors = plt.cm.rainbow(np.linspace(0, 1, len(main_blocks) if main_blocks else 1))
        
        # Dessiner les MainZone avec la palette de couleurs
        for block, color in zip(main_blocks, main_colors):
            x, y, w, h = block['bbox']
            
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Afficher les informations du bloc
            plt.text(x, y-10, 
                    f"Type: {block['label']}\nLines: {block['n_lines']}",
                    color='white',
                    bbox=dict(facecolor='black', alpha=0.7),
                    fontsize=8)
        
        # Dessiner les blocs non-MainZone en gris
        for block in other_blocks:
            x, y, w, h = block['bbox']
            
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2,
                edgecolor='gray',
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Afficher les informations du bloc
            plt.text(x, y-10, 
                    f"Type: {block['label']}\nLines: {block['n_lines']}",
                    color='white',
                    bbox=dict(facecolor='black', alpha=0.7),
                    fontsize=8)
        
        plt.axis('off')
        plt.title(f"Layout Segmentation - {os.path.basename(self.image_path)}")
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
        else:
            plt.show()
            
        return fig
    
    def draw_lines(self, figsize=(15, 20), save_path=None):
        """
        Visualise la segmentation en lignes.
        
        Args:
            figsize: Taille de la figure (width, height)
            save_path: Chemin pour sauvegarder l'image (si None, affiche la figure)
            
        Returns:
            Figure matplotlib
        """
        blocks = self.analyze_layout()
        lines = self.analyze_lines()
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.image)
        
        # Dessiner d'abord les blocs avec une faible opacité
        for block in blocks:
            x, y, w, h = block['bbox']
            
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=1,
                edgecolor='gray',
                facecolor='none',
                alpha=0.3
            )
            ax.add_patch(rect)
        
        # Générer des couleurs pour chaque type de bloc
        block_types = list(set(block['label'] for block in blocks))
        color_map = {block_type: plt.cm.tab10(i % 10) for i, block_type in enumerate(block_types)}
        
        # Dessiner les lignes
        for line in lines:
            x, y, w, h = line['bbox']
            block_type = line['block_type']
            color = color_map.get(block_type, 'red')
            
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=1,
                edgecolor=color,
                facecolor='none',
                alpha=0.7
            )
            ax.add_patch(rect)
        
        # Créer une légende pour les types de blocs
        legend_patches = [patches.Patch(color=color_map.get(block_type, 'red'), 
                                        label=block_type) 
                         for block_type in block_types]
        plt.legend(handles=legend_patches, loc='upper right')
        
        plt.axis('off')
        plt.title(f"Line Segmentation - {os.path.basename(self.image_path)}")
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
        else:
            plt.show()
            
        return fig
    
    def visualize(self, output_dir=None):
        """
        Effectue la visualisation en fonction du type spécifié.
        
        Args:
            output_dir: Dossier où sauvegarder les visualisations
            
        Returns:
            True si la visualisation a réussi, False sinon
        """
        if not self.image or not self.root:
            print(f"Error: Could not load data for {self.image_path}")
            return False
        
        try:
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
                # Créer le nom du fichier de sortie
                base_name = os.path.splitext(os.path.basename(self.image_path))[0]
                vis_type = self.visualization_type
                save_path = os.path.join(output_dir, f"{base_name}_{vis_type}.png")
                
                # Effectuer la visualisation
                if self.visualization_type == 'layout':
                    self.draw_layout(save_path=save_path)
                else:  # 'line'
                    self.draw_lines(save_path=save_path)
                    
                print(f"Visualization saved to {save_path}")
            else:
                # Affichage interactif
                if self.visualization_type == 'layout':
                    self.draw_layout()
                else:  # 'line'
                    self.draw_lines()
                    
            return True
            
        except Exception as e:
            print(f"Error during visualization: {e}")
            import traceback
            traceback.print_exc()
            return False

def visualize_folder(img_dir, xml_dir=None, output_dir=None, visualization_type='layout'):
    """
    Traite tous les fichiers ALTO XML d'un dossier et génère des visualisations.
    
    Args:
        xml_dir: Dossier contenant les fichiers ALTO XML
        img_dir: Dossier contenant les images (si différent de xml_dir)
        output_dir: Dossier où sauvegarder les visualisations
        visualization_type: Type de visualisation ('layout' ou 'line')
        
    Returns:
        Nombre de visualisations réussies
    """
    if xml_dir is None:
        xml_dir = img_dir
        
    if output_dir is None:
        output_dir = xml_dir
        
    # Trouver tous les fichiers XML
    xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
    
    if not xml_files:
        print(f"No XML files found in {xml_dir}")
        return 0
        
    # Traiter chaque fichier XML
    success_count = 0
    
    for xml_file in xml_files:
        base_name = os.path.splitext(os.path.basename(xml_file))[0]
        
        # Chercher l'image correspondante
        img_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        img_file = None
        
        for ext in img_extensions:
            potential_img = os.path.join(img_dir, base_name + ext)
            if os.path.exists(potential_img):
                img_file = potential_img
                break
        
        if not img_file:
            print(f"Warning: No image found for {xml_file}")
            continue
        
        # Créer le visualiseur et générer la visualisation
        visualizer = DocumentVisualizer(img_file, xml_file, visualization_type)
        if visualizer.visualize(output_dir):
            success_count += 1
    
    print(f"Processed {len(xml_files)} XML files with {success_count} successful visualizations")
    return success_count