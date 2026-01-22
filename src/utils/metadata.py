import json
import pandas as pd
import tabulate

from pathlib import Path
from typing import Dict, Optional


def load_metadata(document_dir: str) -> Optional[Dict]:
    """
    Load metadata.json from a document directory.
    
    Args:
        document_dir: Path to document directory
        
    Returns:
        Dictionary of metadata or None if not found
    """
    metadata_path = Path(document_dir) / "metadata.json"
    
    if not metadata_path.exists():
        return None
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: Could not load metadata from {metadata_path}: {e}")
        return None


def aggregate_scores_by_metadata(document_scores, metric_keys=['cer', 'wer']):
    """
    Aggregate scores by metadata features.
    
    Args:
        document_scores: List of document score dicts with metadata (prefixed with 'metadata/')
        metric_keys: List of metric keys to aggregate (e.g., ['cer', 'wer'])
        
    Returns:
        Dictionary mapping feature_name -> {
            'aggregated': list of dicts with mean/std/count per value,
            'summary_metrics': dict with per-value metrics in by_{feature}/ category,
            'unweighted_averages': dict with unweighted average across all values
        }
    """
    if not document_scores:
        return {}
    
    # Convert to DataFrame for easier grouping
    df = pd.DataFrame(document_scores)
    
    # Find metadata columns (those starting with 'metadata/')
    metadata_cols = [col for col in df.columns if col.startswith('metadata/')]
    
    if not metadata_cols:
        return {}
    
    # Find metric columns with the score/ prefix
    score_cols = [f'score/{key}' for key in metric_keys if f'score/{key}' in df.columns]
    
    if not score_cols:
        return {}
    
    aggregated = {}
    
    for metadata_col in metadata_cols:
        feature_name = metadata_col.replace('metadata/', '')
        
        # Group by this feature and calculate statistics
        grouped = df.groupby(metadata_col)[score_cols].agg(['mean', 'std', 'count'])
        
        # Also get page counts
        page_counts = df.groupby(metadata_col)['pages'].sum()
        
        # Flatten multi-index columns
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        grouped = grouped.reset_index()
        
        # Add page counts
        grouped['total_pages'] = page_counts.values
        
        # Rename metadata column
        grouped = grouped.rename(columns={metadata_col: feature_name})
        
        # Create summary metrics dict for wandb logging
        summary_metrics = {}
        per_value_means = {metric: [] for metric in metric_keys}
        
        for _, row in grouped.iterrows():
            feature_value = row[feature_name]
            # Sanitize feature_value for use in metric names
            safe_value = str(feature_value).replace(' ', '_').replace('/', '_').replace('-', '_')
            
            for metric_key in metric_keys:
                mean_col = f'score/{metric_key}_mean'
                if mean_col in row and pd.notna(row[mean_col]):
                    mean_value = row[mean_col]
                    # Per-value metric in by_{feature}/ category
                    summary_metrics[f"by_{feature_name}/{safe_value}/{metric_key}"] = mean_value
                    # Collect for unweighted average
                    per_value_means[metric_key].append(mean_value)
        
        # Calculate unweighted averages across all values of this feature
        # Put these in score/ category
        unweighted_averages = {}
        for metric_key in metric_keys:
            if per_value_means[metric_key]:
                unweighted_avg = sum(per_value_means[metric_key]) / len(per_value_means[metric_key])
                unweighted_averages[metric_key] = unweighted_avg
                # Add to summary metrics in score/ with clear naming
                summary_metrics[f"score/avg_{feature_name}_{metric_key}"] = unweighted_avg
        
        aggregated[feature_name] = {
            'aggregated': grouped.to_dict('records'),
            'summary_metrics': summary_metrics,
            'unweighted_averages': unweighted_averages
        }
    
    return aggregated

def create_metadata_stats(document_scores, results_path):
    """
    Create aggregated statistics by metadata features.
    
    Args:
        document_scores: List of document scores with metadata (prefixed with 'metadata/')
        results_path: Path to save CSV files
        
    Returns:
        List of created CSV filenames
    """
    if not document_scores:
        return []
    
    df = pd.DataFrame(document_scores)
    
    # Find metadata columns (those starting with 'metadata/')
    metadata_cols = [col for col in df.columns if col.startswith('metadata/')]
    
    if not metadata_cols:
        return []
    
    # Extract feature names (remove 'metadata/' prefix)
    features = {col: col.replace('metadata/', '') for col in metadata_cols}
    
    print(f"\n📊 Creating metadata statistics for: {', '.join(features.values())}")
    
    # Find CER and WER columns
    cer_col = 'score/cer' if 'score/cer' in df.columns else None
    wer_col = 'score/wer' if 'score/wer' in df.columns else None
    
    if not cer_col and not wer_col:
        print("  ⚠️  No CER/WER metrics found")
        return []
    
    csv_files = []
    
    # For each metadata feature
    for metadata_col, feature_name in features.items():
        agg_dict = {'pages': 'sum'}
        
        if cer_col:
            agg_dict[cer_col] = ['mean', 'std', 'count']
        if wer_col:
            agg_dict[wer_col] = ['mean', 'std', 'count']
        
        # Group by this metadata feature
        grouped = df.groupby(metadata_col).agg(agg_dict)
        
        # Flatten column names
        grouped.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                            for col in grouped.columns.values]
        grouped = grouped.reset_index()
        
        # Rename the metadata column to just the feature name
        grouped = grouped.rename(columns={metadata_col: feature_name})
        
        # Save
        csv_path = results_path / f"scores_by_{feature_name}.csv"
        grouped.to_csv(csv_path, index=False)
        csv_files.append(f"scores_by_{feature_name}.csv")
        
        print(f"  • {feature_name}: {len(grouped)} groups")
            
    
    return csv_files

def display_metadata_metrics(metadata_metrics):
    """
    Display metadata-aggregated metrics in a formatted way.
    
    Args:
        metadata_metrics: Dictionary of metadata-aggregated metrics
    """
    if not metadata_metrics:
        return
    
    print("\n📊 Metrics by Metadata:")
    print("=" * 70)
    
    # Group metrics by feature
    by_feature = {}
    avg_by_feature = {}
    
    for key, value in metadata_metrics.items():
        # Handle averages in score/
        if key.startswith('score/avg_'):
            # Format: "score/avg_{feature}_{metric}"
            key_without_prefix = key[10:]  # Remove 'score/avg_'
            parts = key_without_prefix.rsplit('_', 1)  # Split on last underscore
            
            if len(parts) == 2:
                feature, metric = parts
                if feature not in avg_by_feature:
                    avg_by_feature[feature] = {}
                avg_by_feature[feature][metric] = value
        
        # Handle per-value metrics in by_{feature}/
        elif key.startswith('by_'):
            # Format: "by_{feature}/{value}/{metric}"
            key_without_prefix = key[3:]  # Remove 'by_'
            parts = key_without_prefix.split('/')
            
            if len(parts) == 3:
                feature, feature_value, metric = parts
                
                if feature not in by_feature:
                    by_feature[feature] = {}
                if feature_value not in by_feature[feature]:
                    by_feature[feature][feature_value] = {}
                
                by_feature[feature][feature_value][metric] = value
    
    # Display by feature
    for feature in sorted(by_feature.keys()):
        values = by_feature[feature]
        print(f"\n  {feature.upper()}:")
        
        table_data = [["Value", "CER", "WER"]]
        for value_name, metrics in sorted(values.items()):
            cer = metrics.get('cer', 'N/A')
            wer = metrics.get('wer', 'N/A')
            
            if isinstance(cer, float):
                cer = f"{cer:.4f}"
            if isinstance(wer, float):
                wer = f"{wer:.4f}"
            
            table_data.append([value_name, cer, wer])
        
        # Add unweighted average row
        if feature in avg_by_feature:
            avg_metrics = avg_by_feature[feature]
            avg_cer = avg_metrics.get('cer', 'N/A')
            avg_wer = avg_metrics.get('wer', 'N/A')
            
            if isinstance(avg_cer, float):
                avg_cer = f"{avg_cer:.4f}"
            if isinstance(avg_wer, float):
                avg_wer = f"{avg_wer:.4f}"
            
            # Add separator
            table_data.append(["---", "---", "---"])
            table_data.append([f"Avg (unweighted)", avg_cer, avg_wer])
        
        print(tabulate.tabulate(table_data, headers="firstrow", tablefmt="simple"))