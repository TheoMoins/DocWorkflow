#!/usr/bin/env python3
"""
Script de debug pour analyser pourquoi le CER est si élevé
alors que les textes semblent identiques visuellement.
"""

from lxml import etree as ET
from jiwer import cer
import difflib
import os

def extract_text_from_alto(alto_path):
    """Extrait le texte comme le fait base_htr.py"""
    tree = ET.parse(alto_path)
    root = tree.getroot()
    ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
    
    # Extract all String elements
    strings = root.findall('.//alto:String', ns)
    texts = [s.get('CONTENT', '') for s in strings if s.get('CONTENT')]
    
    return ' '.join(texts) if texts else ''

def extract_text_line_by_line(alto_path):
    """Extrait le texte ligne par ligne pour voir la structure"""
    tree = ET.parse(alto_path)
    root = tree.getroot()
    ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
    
    lines = []
    for textline in root.findall('.//alto:TextLine', ns):
        strings = textline.findall('.//alto:String', ns)
        if strings:
            line_text = ' '.join([s.get('CONTENT', '') for s in strings if s.get('CONTENT')])
            if line_text:
                lines.append(line_text)
    
    return lines

def analyze_files(gt_path, pred_path):
    """Analyse détaillée des différences"""
    
    print("=" * 80)
    print("ANALYSE DÉTAILLÉE DES DIFFÉRENCES")
    print("=" * 80)
    
    # 1. Extraire les textes comme le fait le système
    gt_text = extract_text_from_alto(gt_path)
    pred_text = extract_text_from_alto(pred_path)
    
    print(f"\n📄 Ground Truth: {os.path.basename(gt_path)}")
    print(f"📄 Prediction:   {os.path.basename(pred_path)}")
    
    # 2. Statistiques de base
    print(f"\n📊 STATISTIQUES:")
    print(f"  GT   - Longueur: {len(gt_text):,} caractères, {len(gt_text.split()):,} mots")
    print(f"  PRED - Longueur: {len(pred_text):,} caractères, {len(pred_text.split()):,} mots")
    print(f"  Ratio: {len(pred_text)/len(gt_text):.2f}x")
    
    # 3. CER
    cer_score = cer([gt_text], [pred_text])
    print(f"\n📈 CER: {cer_score:.4f} ({cer_score*100:.2f}%)")
    
    # 4. Afficher les premiers et derniers caractères
    print(f"\n🔍 PREMIERS 200 CARACTÈRES:")
    print(f"  GT:   {repr(gt_text[:200])}")
    print(f"  PRED: {repr(pred_text[:200])}")
    
    print(f"\n🔍 DERNIERS 200 CARACTÈRES:")
    print(f"  GT:   {repr(gt_text[-200:])}")
    print(f"  PRED: {repr(pred_text[-200:])}")
    
    # 5. Lignes
    gt_lines = extract_text_line_by_line(gt_path)
    pred_lines = extract_text_line_by_line(pred_path)
    
    print(f"\n📝 NOMBRE DE LIGNES:")
    print(f"  GT:   {len(gt_lines)}")
    print(f"  PRED: {len(pred_lines)}")
    
    # 6. Comparer ligne par ligne
    print(f"\n📋 COMPARAISON DES PREMIÈRES LIGNES:")
    for i in range(min(10, len(gt_lines), len(pred_lines))):
        gt_line = gt_lines[i] if i < len(gt_lines) else "[MANQUANT]"
        pred_line = pred_lines[i] if i < len(pred_lines) else "[MANQUANT]"
        
        match = "✓" if gt_line == pred_line else "✗"
        print(f"\n  Ligne {i+1} {match}")
        print(f"    GT:   {gt_line[:80]}")
        print(f"    PRED: {pred_line[:80]}")
        
        if gt_line != pred_line and i < 5:
            # Afficher les différences
            diff = list(difflib.unified_diff(
                gt_line.split(), 
                pred_line.split(), 
                lineterm='', 
                n=0
            ))
            if len(diff) > 2:
                print(f"    DIFF: {' '.join(diff[2:])}")
    
    # 7. Chercher des duplications
    print(f"\n🔎 VÉRIFICATION DES DUPLICATIONS:")
    
    # Chercher si le texte de prédiction contient des répétitions
    lines_text = '\n'.join(pred_lines)
    
    # Compter les lignes dupliquées
    seen = {}
    duplicates = []
    for i, line in enumerate(pred_lines):
        if line in seen and line.strip():
            duplicates.append((i, line, seen[line]))
        else:
            seen[line] = i
    
    if duplicates:
        print(f"  ⚠️  Trouvé {len(duplicates)} lignes dupliquées dans la prédiction!")
        for i, line, first_seen in duplicates[:5]:
            print(f"    Ligne {i+1} (identique à ligne {first_seen+1}): {line[:60]}...")
    else:
        print(f"  ✓ Aucune duplication évidente")
    
    # 8. Vérifier les espaces et caractères invisibles
    print(f"\n🔎 CARACTÈRES SPÉCIAUX:")
    gt_spaces = gt_text.count(' ')
    pred_spaces = pred_text.count(' ')
    gt_newlines = gt_text.count('\n')
    pred_newlines = pred_text.count('\n')
    
    print(f"  GT   - Espaces: {gt_spaces:,}, Retours à la ligne: {gt_newlines}")
    print(f"  PRED - Espaces: {pred_spaces:,}, Retours à la ligne: {pred_newlines}")
    
    # 9. Sauvegarder les textes pour inspection manuelle
    output_dir = "/home/claude/"
    
    with open(os.path.join(output_dir, "gt_text.txt"), "w", encoding="utf-8") as f:
        f.write(gt_text)
    
    with open(os.path.join(output_dir, "pred_text.txt"), "w", encoding="utf-8") as f:
        f.write(pred_text)
    
    with open(os.path.join(output_dir, "gt_lines.txt"), "w", encoding="utf-8") as f:
        f.write('\n'.join(gt_lines))
    
    with open(os.path.join(output_dir, "pred_lines.txt"), "w", encoding="utf-8") as f:
        f.write('\n'.join(pred_lines))
    
    print(f"\n💾 Textes sauvegardés dans {output_dir}")
    print(f"  - gt_text.txt")
    print(f"  - pred_text.txt")
    print(f"  - gt_lines.txt")
    print(f"  - pred_lines.txt")
    
    # 10. Analyse diff complète
    print(f"\n📄 GÉNÉRATION DU DIFF COMPLET...")
    diff_output = '\n'.join(difflib.unified_diff(
        gt_text.splitlines(keepends=True),
        pred_text.splitlines(keepends=True),
        fromfile='ground_truth',
        tofile='prediction',
        lineterm=''
    ))
    
    with open(os.path.join(output_dir, "full_diff.txt"), "w", encoding="utf-8") as f:
        f.write(diff_output)
    
    print(f"  - full_diff.txt")
    
    print("\n" + "=" * 80)
    print("ANALYSE TERMINÉE")
    print("=" * 80)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python debug_cer_issue.py <gt_file.xml> <pred_file.xml>")
        print("\nExemple:")
        print("  python debug_cer_issue.py \\")
        print("    /path/to/gt/btv1b84526412_f22.xml \\")
        print("    /path/to/pred/btv1b84526412_f22.xml")
        sys.exit(1)
    
    gt_path = sys.argv[1]
    pred_path = sys.argv[2]
    
    if not os.path.exists(gt_path):
        print(f"❌ Fichier GT introuvable: {gt_path}")
        sys.exit(1)
    
    if not os.path.exists(pred_path):
        print(f"❌ Fichier de prédiction introuvable: {pred_path}")
        sys.exit(1)
    
    analyze_files(gt_path, pred_path)