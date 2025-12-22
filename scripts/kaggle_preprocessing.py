import glob
import pandas as pd
from pathlib import Path
from lxml import etree as ET
import jiwer


def extract_text_from_alto(alto_path):
    """Extract full page text from ALTO."""
    tree = ET.parse(alto_path)
    root = tree.getroot()
    ns = {'alto': 
          'http://www.loc.gov/standards/alto/ns-v4#'}
    
    strings = root.findall('.//alto:String', ns)
    texts = [s.get('CONTENT', '') for s in strings if s.get('CONTENT')]
    
    return ' '.join(texts)

def create_submission_csv(data_dir, output_csv):
    """Create CSV format: image_id, text"""
    xml_files = sorted(glob.glob(f"{data_dir}/*.xml"))
    
    data = []
    for xml_path in xml_files:
        image_id = Path(xml_path).stem
        text = extract_text_from_alto(xml_path)
        data.append({'image_id': image_id, 'text': text})
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Created {output_csv} with {len(df)} entries")

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Kaggle evaluation function.
    
    Args:
        solution: DataFrame with columns [image_id, text]
        submission: DataFrame with columns [image_id, text]
        row_id_column_name: Name of ID column ('image_id')
    
    Returns:
        Score (lower is better for CER)
    """
    
    # Vérifier que toutes les images sont présentes
    if not set(submission[row_id_column_name]) == set(solution[row_id_column_name]):
        missing = set(solution[row_id_column_name]) - set(submission[row_id_column_name])
        raise ValueError(f"Missing predictions for images: {missing}")
    
    # Aligner les DataFrames
    solution = solution.sort_values(row_id_column_name).reset_index(drop=True)
    submission = submission.sort_values(row_id_column_name).reset_index(drop=True)
    
    # Extraire les textes
    ground_truth = solution['text'].tolist()
    predictions = submission['text'].tolist()
    
    # Calculer CER (Character Error Rate)
    cer_score = jiwer.cer(ground_truth, predictions)
    
    # Optionnel : calculer d'autres métriques pour le leaderboard détaillé
    wer_score = jiwer.wer(ground_truth, predictions)
    
    print(f"CER: {cer_score:.4f}")
    print(f"WER: {wer_score:.4f}")
    
    # Retourner le score principal (CER)
    return cer_score

if __name__ == "__main__":
    # Créer le fichier solution (secret)
    create_submission_csv("../data/HTRomance-french/data/test", "public_test.csv")
    
