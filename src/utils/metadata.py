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