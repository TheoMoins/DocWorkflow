import wandb
from pathlib import Path
from src.utils.metadata import aggregate_scores_by_metadata

def _clean_value_for_wandb(value):
    """
    Clean a value to be compatible with wandb tables.
    Converts empty strings to None for numeric fields.
    
    Args:
        value: Value to clean
        
    Returns:
        Cleaned value (None if empty string, otherwise unchanged)
    """
    if value == '':
        return None
    return value


def prepare_wandb_data(page_scores, document_scores, structure_type, results_dir):
    """
    Prepare tables and files for wandb logging.
    
    Args:
        page_scores: List of page-level scores
        document_scores: List of document-level scores (or None)
        structure_type: 'flat' or 'hierarchical'
        results_dir: Directory where CSV files are saved
        
    Returns:
        Tuple of (tables_dict, files_dict, metadata_metrics_dict)
    """
    
    tables = {}
    files = {}
    metadata_metrics = {}
    
    if not page_scores:
        return tables, files, metadata_metrics
    
    # Page scores table
    if page_scores:
        columns = list(page_scores[0].keys())
        # Clean data: convert empty strings to None
        data = [[_clean_value_for_wandb(score.get(col, None)) for col in columns] 
                for score in page_scores]
        tables['page_scores'] = wandb.Table(columns=columns, data=data)
    
    # Document scores table (for hierarchical)
    if document_scores:
        columns = list(document_scores[0].keys())
        # Clean data: convert empty strings to None
        data = [[_clean_value_for_wandb(score.get(col, None)) for col in columns] 
                for score in document_scores]
        tables['document_scores'] = wandb.Table(columns=columns, data=data)
        
        # Check for metadata columns
        metadata_cols = [col for col in columns if col.startswith('metadata/')]
        if metadata_cols:
            # Aggregate by metadata
            metadata_aggregations = aggregate_scores_by_metadata(
                document_scores, 
                metric_keys=['cer', 'wer']
            )
            
            # Create tables for each metadata feature
            for feature_name, feature_data in metadata_aggregations.items():
                # Create wandb table
                agg_data = feature_data['aggregated']
                if agg_data:
                    agg_columns = list(agg_data[0].keys())
                    agg_rows = [[_clean_value_for_wandb(row.get(col, None)) 
                                for col in agg_columns] 
                               for row in agg_data]
                    tables[f'by_{feature_name}'] = wandb.Table(
                        columns=agg_columns, 
                        data=agg_rows
                    )
                
                # Add summary metrics for logging
                metadata_metrics.update(feature_data['summary_metrics'])
                
                # Add CSV file if it exists
                csv_path = results_dir / f"scores_by_{feature_name}.csv"
                if csv_path.exists():
                    files[f"scores_by_{feature_name}"] = str(csv_path)
    
    # Add main CSV files
    if structure_type == 'hierarchical':
        if (results_dir / "scores_per_document.csv").exists():
            files["scores_per_document"] = str(results_dir / "scores_per_document.csv")
        if (results_dir / "scores_all_pages.csv").exists():
            files["scores_all_pages"] = str(results_dir / "scores_all_pages.csv")
    else:
        if (results_dir / "scores_per_page.csv").exists():
            files["scores_per_page"] = str(results_dir / "scores_per_page.csv")
    
    return tables, files, metadata_metrics