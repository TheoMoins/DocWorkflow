import pandas as pd
from pathlib import Path

import jiwer
from jiwer import cer, wer


def aggregate_metrics(metrics_list):
    """
    Aggregate metrics from multiple batches.
    Default: average numeric values.
    Override for task-specific aggregation.
    """
    if not metrics_list:
        return {}
    if len(metrics_list) == 1:
        return metrics_list[0]
    
    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())
    
    result = {}
    for key in all_keys:
        values = [m[key] for m in metrics_list if key in m and isinstance(m[key], (int, float))]
        if values:
            if key.startswith('total/'):
                result[key] = sum(values)  # Sum totals
            else:
                result[key] = sum(values) / len(values)  # Average metrics
    
    return result



def calculate_htr_metrics(all_gt_texts, all_pred_texts, page_scores):
    """
    Calculate HTR-specific metrics from texts.
    
    Args:
        all_gt_texts: List of ground truth text strings
        all_pred_texts: List of predicted text strings
        page_scores: List of per-page score dictionaries
        
    Returns:
        Dictionary of HTR metrics
    """
    cer_score = cer(all_gt_texts, all_pred_texts)
    wer_score = wer(all_gt_texts, all_pred_texts)
    
    char_accuracy = 1.0 - cer_score
    word_accuracy = 1.0 - wer_score
    
    # Get detailed error counts
    cer_output = jiwer.process_characters(all_gt_texts, all_pred_texts)
    wer_output = jiwer.process_words(all_gt_texts, all_pred_texts)
    
    metrics_dict = {
        "score/cer": cer_score,
        "score/wer": wer_score,
        "accuracy/char_accuracy": char_accuracy,
        "accuracy/word_accuracy": word_accuracy,
        "total/total_chars": sum(len(text) for text in all_pred_texts),
        "total/total_words": sum(len(text.split()) for text in all_pred_texts),
        "detailed/char_insertions": cer_output.insertions,
        "detailed/char_deletions": cer_output.deletions,
        "detailed/char_substitutions": cer_output.substitutions,
        "detailed/word_insertions": wer_output.insertions,
        "detailed/word_deletions": wer_output.deletions,
        "detailed/word_substitutions": wer_output.substitutions,
    }
    
    # Add worst pages
    if page_scores:
        worst_pages = sorted(page_scores, key=lambda x: x['cer'], reverse=True)[:5]
        for i, page in enumerate(worst_pages, 1):
            metrics_dict[f"worst/top{i}_file"] = page['page']
            metrics_dict[f"worst/top{i}_cer"] = page['cer']
    
    return metrics_dict


def save_score_csvs(results_dir, page_scores, document_scores=None, structure_type='flat'):
    """
    Save detailed score CSV files.
    
    Args:
        results_dir: Directory to save CSV files
        page_scores: List of per-page score dictionaries
        document_scores: List of per-document score dictionaries (None for flat)
        structure_type: 'flat' or 'hierarchical'
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    if not page_scores:
        return
    
    if structure_type == 'flat':
        # Single per-page CSV
        df = pd.DataFrame(page_scores)
        csv_path = results_path / "scores_per_page.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved: {csv_path}")
        
    else:  # hierarchical
        # Per-document CSVs
        documents = set(s['document'] for s in page_scores)
        
        for doc in documents:
            doc_scores = [s for s in page_scores if s['document'] == doc]
            df = pd.DataFrame(doc_scores)
            # Remove 'document' column for individual CSVs
            if 'document' in df.columns:
                df = df.drop('document', axis=1)
            
            doc_path = results_path / doc
            doc_path.mkdir(exist_ok=True)
            df.to_csv(doc_path / "scores_per_page.csv", index=False)
        
        # Global CSVs
        if document_scores:
            pd.DataFrame(document_scores).to_csv(
                results_path / "scores_per_document.csv", index=False
            )
        
        pd.DataFrame(page_scores).to_csv(
            results_path / "scores_all_pages.csv", index=False
        )
        
        print(f"\n✓ Saved: scores_per_document.csv, scores_all_pages.csv, "
              f"and {len(documents)} per-document CSVs")