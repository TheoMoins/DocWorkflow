import json
import pandas as pd

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
        document_scores: List of document score dicts with metadata
        metric_keys: List of metric keys to aggregate (e.g., ['cer', 'wer'])
        
    Returns:
        Dictionary mapping feature_name -> feature_value -> aggregated_metrics
    """
    
    # Convert to DataFrame for easier grouping
    df = pd.DataFrame(document_scores)
    
    # Find metadata columns (exclude standard columns)
    standard_cols = {'document', 'pages'} | set(metric_keys)
    metadata_cols = [col for col in df.columns if col not in standard_cols]
    
    if not metadata_cols:
        return {}
    
    aggregated = {}
    
    for feature in metadata_cols:
        # Group by this feature and calculate means
        grouped = df.groupby(feature)[metric_keys].agg(['mean', 'std', 'count'])
        
        # Flatten multi-index columns
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        grouped = grouped.reset_index()
        
        aggregated[feature] = grouped.to_dict('records')
    
    return aggregated

def create_metadata_stats(document_scores, results_path):
    """
    Create aggregated statistics by metadata features.
    
    Args:
        document_scores: List of document scores with metadata
        results_path: Path to save CSV files
        
    Returns:
        List of created CSV filenames
    """    
    # Detect if there's metadata (columns beyond standard ones)
    if not document_scores:
        return []
    
    df = pd.DataFrame(document_scores)
    standard_cols = {'document', 'pages'}
    
    # Find metric columns (those starting with known prefixes)
    metric_prefixes = ['score/', 'dataset_test/', 'accuracy/', 'total/', 'detailed/']
    metric_cols = [col for col in df.columns 
                   if any(col.startswith(prefix) for prefix in metric_prefixes)]
    
    # Extract just the metric name (e.g., 'score/cer' -> 'cer')
    simple_metric_names = []
    for col in metric_cols:
        for prefix in metric_prefixes:
            if col.startswith(prefix):
                simple_name = col[len(prefix):]
                simple_metric_names.append(simple_name)
                break
    
    # Find metadata columns
    metadata_cols = [col for col in df.columns 
                     if col not in standard_cols and col not in metric_cols]
    
    if not metadata_cols:
        return []
    
    print(f"\n📊 Creating metadata statistics for: {', '.join(metadata_cols)}")
    
    csv_files = []
    
    # For each metadata feature, create aggregated stats
    for feature in metadata_cols:
        # Group by this feature
        feature_df = df.groupby(feature).agg({
            'pages': 'sum',  # Total pages per group
            **{col: ['mean', 'std', 'count'] for col in metric_cols}
        })
        
        # Flatten multi-index columns
        feature_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                              for col in feature_df.columns.values]
        feature_df = feature_df.reset_index()
        
        # Save to CSV
        csv_filename = f"scores_by_{feature}.csv"
        csv_path = results_path / csv_filename
        feature_df.to_csv(csv_path, index=False)
        csv_files.append(csv_filename)
        
        # Display summary
        print(f"  • {feature}: {len(feature_df)} groups")
    
    return csv_files