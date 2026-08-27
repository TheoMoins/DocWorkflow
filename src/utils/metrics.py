import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import jiwer
from jiwer import cer, wer
from jiwer.transforms import AbstractTransform
import regex
from pathlib import Path

from torchmetrics.text import CharErrorRate, WordErrorRate
import unicodedata
from collections import Counter
from itertools import chain

from src.utils.metadata import create_metadata_stats


def aggregate_metrics(metrics_list):
    """
    Aggregate metrics from multiple batches.
    Default: average numeric values.
    Override for task-specific aggregation.

    Note on aggregation (hierarchical datasets):
        Rate metrics (CER, WER, ...) are aggregated as an **unweighted mean of
        the per-document values** (macro-average): each document counts equally,
        regardless of its number of characters. This is deliberate — it prevents
        a few very long documents from dominating the reported score — but it
        differs from the micro-average (character-weighted) convention usual in
        HTR, so the two are not directly comparable.

        Only `total/` keys are summed rather than averaged.
    """
    if not metrics_list:
        return {}
    if len(metrics_list) == 1:
        return metrics_list[0]
    

    all_keys = dict.fromkeys(k for m in metrics_list for k in m)
    NUMERIC = (int, float, np.integer, np.floating)

    result = {}
    for key in all_keys:
        values = [float(m[key]) for m in metrics_list
                  if key in m and isinstance(m[key], NUMERIC) and not isinstance(m[key], bool)]
        if values:
            #TODO this should include detailed/
            if key.startswith('total/'):
                result[key] = sum(values)  # Sum totals
            else:
                result[key] = sum(values) / len(values)  # Average metrics
        else:
            if key.startswith('analysis/'):
                #print(key)
                all_dicts = [m[key] for m in metrics_list if key in m]
                #print(all_dicts)
                value = tuple(dict(Counter(chain.from_iterable([d[i] for d in all_dicts]))) for i in range(3))
                result[key] = value
    
    return result



def calculate_htr_metrics(all_gt_texts, all_pred_texts, page_scores,
                          competition_preds=None, competition_gt=None):
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
    # Need some custom work to make jiwer process diacritics correctly
    class DiacriticTransform(AbstractTransform):
        def process_string(self, s: str):
            s = regex.findall(r'\X', unicodedata.normalize("NFD", s))
            return s
    
    dt = DiacriticTransform()
    # use process_words because we force it to treat each combined character (NOT unicode codepoint) as a character
    cer_output = jiwer.process_words(all_gt_texts, all_pred_texts, reference_transform=dt, hypothesis_transform=dt)
    #TODO: normalize wer inputs as well while keeping them as words
    wer_output = jiwer.process_words(all_gt_texts, all_pred_texts)

    #cer_analysis = jiwer.visualize_error_counts(cer_output)
    #wer_analysis = jiwer.visualize_error_counts(wer_output)

    cer_analysis = jiwer.collect_error_counts(cer_output)
    wer_analysis = jiwer.collect_error_counts(wer_output)

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
        "analysis/cer": cer_analysis,
        "analysis/wer": wer_analysis
    }
    
    # Add worst pages
    if page_scores:
        worst_pages = sorted(page_scores, key=lambda x: x['cer'], reverse=True)[:5]
        for i in range(1, 6):  # Top 5
            if i <= len(worst_pages):
                page_info = worst_pages[i-1]
                
                # Format: "document/page" if document exists, else just "page"
                if 'document' in page_info and page_info['document']:
                    full_name = f"{page_info['document']}/{page_info['page']}"
                else:
                    full_name = page_info['page']
                
                metrics_dict[f"worst/top{i}_file"] = full_name
                metrics_dict[f"worst/top{i}_cer"] = page_info['cer']
            else:
                metrics_dict[f"worst/top{i}_file"] = None
                metrics_dict[f"worst/top{i}_cer"] = None

    # Competition metrics (CMMHWR: line-level by ID, torchmetrics, NFD)
    if competition_preds is not None and competition_gt is not None:

        common_ids = set(competition_preds.keys()) & set(competition_gt.keys())
        if common_ids:
            comp_cer = CharErrorRate()
            comp_wer = WordErrorRate()
            for line_id in common_ids:
                pred = unicodedata.normalize('NFD', competition_preds[line_id])
                gt = unicodedata.normalize('NFD', competition_gt[line_id])
                comp_cer(pred, gt)
                comp_wer(pred, gt)
            metrics_dict["competition/cer"] = comp_cer.compute().item()
            metrics_dict["competition/wer"] = comp_wer.compute().item()
            metrics_dict["competition/matched_lines"] = len(common_ids)
        else:
            metrics_dict["competition/cer"] = None
            metrics_dict["competition/wer"] = None
            metrics_dict["competition/matched_lines"] = 0
    
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
        csv_files = []
        
        if document_scores:
            doc_df = pd.DataFrame(document_scores)
            doc_csv = results_path / "scores_per_document.csv"
            doc_df.to_csv(doc_csv, index=False)
            csv_files.append("scores_per_document.csv")
            
            # Check if metadata is present and create aggregated stats
            metadata_stats = create_metadata_stats(document_scores, results_path)
            if metadata_stats:
                csv_files.extend(metadata_stats)
        
        all_csv = results_path / "scores_all_pages.csv"
        pd.DataFrame(page_scores).to_csv(all_csv, index=False)
        csv_files.append("scores_all_pages.csv")
        
        print(f"\n✓ Saved: {', '.join(csv_files)}, "
              f"and {len(documents)} per-document CSVs")

def save_zonemap_csv(results_dir: Path, zonemap_metrics: dict):
    """Save ZoneMap metrics in a structured, readable CSV."""
    detection_keys = ['score', 'match', 'miss', 'false_alarm', 'split', 'merge', 'multiple']
    count_keys = ['n_match', 'n_miss', 'n_false_alarm', 'n_split', 'n_merge', 'n_multiple']
    recognition_keys = [
        'char_precision', 'char_recall', 'char_f1',
        'word_precision', 'word_recall', 'word_f1',
    ]

    rows = []

    rows.append(('--- Detection (area-based) ---', '', ''))
    rows.append(('zonemap/score', zonemap_metrics.get('zonemap/score', ''), 'Error % (lower=better)'))
    for k in detection_keys[1:]:
        val = zonemap_metrics.get(f'zonemap/{k}', '')
        rows.append((f'zonemap/{k}', val, 'Fraction of GT area'))

    rows.append(('--- Detection (counts per doc) ---', '', ''))
    for k in count_keys:
        val = zonemap_metrics.get(f'zonemap/{k}', '')
        rows.append((f'zonemap/{k}', val, 'Avg group count'))

    has_recognition = any(f'zonemap/{k}' in zonemap_metrics for k in recognition_keys)
    if has_recognition:
        rows.append(('--- Recognition (ZoneMapAltCnt) ---', '', ''))
        for k in recognition_keys:
            val = zonemap_metrics.get(f'zonemap/{k}', '')
            if 'char' in k:
                label = 'Char-level'
            elif 'f1' in k:
                label = 'Word-level F1'
            else:
                label = 'Word-level'
            rows.append((f'zonemap/{k}', val, label))

    df = pd.DataFrame(rows, columns=['metric', 'value', 'description'])
    out = results_dir / 'results_zonemap.csv'
    df.to_csv(out, index=False)
    print(f"ZoneMap results saved to {out}")


def save_zonemap_scatter(results_dir: Path, document_scores: list, global_zonemap: dict):
    """Scatter plot: ZoneMap score (x) vs char F1 (y), one point per document + global."""
    BG      = '#f5f5f5'
    GRID    = '#36364a'
    DOC_C   = '#0b38a3'
    GLOB_C  = '#ff7043'
    TEXT_C  = '#111829'
    LABEL_C = '#0b38a3'

    docs = [d for d in document_scores
            if d.get('zonemap/score') is not None and d.get('zonemap/char_f1') is not None]
    if not docs:
        return

    xs = np.array([100-d['zonemap/score']   for d in docs], dtype=float)
    ys = np.array([d['zonemap/char_f1'] for d in docs], dtype=float)
    names = [d['document'] for d in docs]

    gx = 100-global_zonemap.get('zonemap/score')
    gy = global_zonemap.get('zonemap/char_f1')

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Median reference lines
    # ax.axvline(np.median(xs), linestyle='--', color=GRID, linewidth=1, zorder=1)
    # ax.axhline(np.median(ys), linestyle='--', color=GRID, linewidth=1, zorder=1)

    ax.grid(True, linestyle='--', color=GRID, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Document points
    ax.scatter(xs, ys, color=DOC_C, s=65, alpha=0.85, zorder=3, linewidths=0)
    for x, y, name in zip(xs, ys, names):
        ax.annotate(name, (x, y),
                    textcoords='offset points', xytext=(6, 3),
                    fontsize=6.5, color=LABEL_C, zorder=4)

    # Global aggregate point
    if gx is not None and gy is not None:
        ax.scatter([gx], [gy], color=GLOB_C, s=130, zorder=5,
                   marker='D', linewidths=0)
        ax.annotate('global', (gx, gy),
                    textcoords='offset points', xytext=(7, 4),
                    fontsize=8.5, color=GLOB_C, fontweight='bold', zorder=6)

    ax.set_xlabel('ZoneMap score',
                  color=TEXT_C, fontsize=9, labelpad=8)
    ax.set_ylabel('Character F1',
                  color=TEXT_C, fontsize=9, labelpad=8)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.tick_params(colors=TEXT_C, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    plt.tight_layout()
    out = results_dir / 'zonemap_scatter.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"ZoneMap scatter plot saved to {out}")