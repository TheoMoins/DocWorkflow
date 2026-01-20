from abc import ABC, abstractmethod
import torch
import wandb
from datetime import datetime
import tabulate
from pathlib import Path
import os

from src.utils.visualisation import visualize_folder
from src.utils.dataset_structure import discover_dataset_structure
from src.utils.metrics import aggregate_metrics, save_score_csvs

class BaseTask(ABC):
    """
    Abstract base class for all document analysis models.
    """
    
    def __init__(self, config):
        """
        Initialize the model with its configuration.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.name = config.get('run_name', "unknown")
        self.model = None
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.use_wandb = config.get('use_wandb', True)
        self.wandb_project = config.get('wandb_project', None)
               
    def to_device(self, device=None):
        """
        Move the model to the specified device.
        
        Args:
            device: Target device ('cuda', 'cpu', etc.)
        """
        if device:
            self.device = device
        if self.model:
            self.model.to(self.device)
    
    
    def _init_wandb(self):
        """
        Initialize a Weights & Biases session.
        """
        if not self.use_wandb:
            return None
        
        # Load API key from file if it exists
        api_key_file = Path("wandb_api_key.txt")
        if api_key_file.exists():
            try:
                with open(api_key_file, 'r') as f:
                    api_key = f.read().strip()
                    if api_key:
                        wandb.login(key=api_key, relogin=True)
            except Exception as e:
                print(f"Warning: Could not load wandb API key from {api_key_file}: {e}")
        

        return wandb.init(
            project=self.wandb_project, 
            name=f"eval-{self.model_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=self.config
        )
    
    def _log_to_wandb(self, metrics, run=None):
        """
        Log metrics to Weights & Biases.
        """
        if not self.use_wandb:
            return
    
        if run:
            run.log(metrics)
        else:
            wandb.log(metrics)
    
    def _finish_wandb(self, run=None):
        """
        Finish a wandb session.
        """
        if not self.use_wandb:
            return 

        if run:
            run.finish()
        else:
            wandb.finish()

    @abstractmethod
    def load(self):
        """Load the model from a weights file."""
        pass
    
    @abstractmethod
    def train(self, **kwargs):
        """
        Train the model.
        
        Args:
            **kwargs: Additional training arguments
        """
        pass
    
    def _display_metrics(self, metrics):
        """
        Display evaluation metrics in a formatted way.
        
        Args:
            metrics: Dictionary of evaluation metrics
        """
        table = [["Metric", "Value"]]
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                formatted_value = round(value, 4)
            else:
                formatted_value = value
            
            table.append([key, formatted_value])
        
        print("\nEvaluation Results:")
        print(tabulate.tabulate(table, tablefmt="grid", headers="firstrow"))
    

    @abstractmethod
    def _process_batch(self, file_paths, source_dir, output_dir, **kwargs):
        """
        Process a batch of files (images or ALTO XMLs).
        
        This is the method that each task must implement with its specific logic.
        
        Args:
            file_paths: List of file paths to process
            source_dir: Source directory (for finding related files like XML)
            output_dir: Output directory for results
            **kwargs: Additional task-specific arguments
            
        Returns:
            Results from processing (task-specific format)
        """
        pass

    @abstractmethod
    def _score_batch(self, pred_files, gt_files, pred_dir, gt_dir):
        """
        Score a batch of predictions against ground truth.
        Must be implemented by subclasses.
        
        Args:
            pred_files: List of prediction file paths
            gt_files: List of ground truth file paths (same order)
            pred_dir: Prediction directory (for context)
            gt_dir: Ground truth directory (for context)
            
        Returns:
            Tuple of (metrics_dict, page_scores_list)
        """
        pass

    def _score_hierarchical(self, pred_path, gt_path):
        """
        Score a hierarchical dataset structure.
        Calls _score_batch for each subdirectory.
        """        
        structure_info = discover_dataset_structure(gt_path, self._get_score_file_extensions())
        
        all_page_scores = []
        document_scores = []
        all_metrics_for_aggregation = []
        
        total_subdirs = len(structure_info['subdirs'])
        
        for idx, (subdir_path, gt_files) in enumerate(structure_info['structure'].items(), start=1):
            subdir_name = Path(subdir_path).name
            pred_subdir = Path(pred_path) / subdir_name
            
            print(f"[{idx:02d}/{total_subdirs:02d}] 📊 {subdir_name}")
            
            if not pred_subdir.exists():
                print(f"  ⚠️  No predictions found")
                continue
            
            # Find matching prediction files
            pred_files = []
            matched_gt_files = []
            for gt_file in gt_files:
                pred_file = pred_subdir / Path(gt_file).name
                if pred_file.exists():
                    pred_files.append(str(pred_file))
                    matched_gt_files.append(gt_file)
            
            if not pred_files:
                print(f"  ⚠️  No matching predictions")
                continue
            
            # Score this subdirectory
            doc_metrics, doc_page_scores = self._score_batch(
                pred_files, matched_gt_files,
                str(pred_subdir), str(subdir_path)
            )
            
            # Add document info to page scores
            for score in doc_page_scores:
                score['document'] = subdir_name
            
            all_page_scores.extend(doc_page_scores)
            all_metrics_for_aggregation.append(doc_metrics)
            
            # Document summary: just add document name and page count to metrics
            doc_summary = {
                'document': subdir_name,
                'pages': len(doc_page_scores),
                **doc_metrics  # Copier toutes les métriques
            }
            
            document_scores.append(doc_summary)
            
            print(f"  ✓ {len(doc_page_scores)} pages scored")

        
        # Aggregate metrics
        global_metrics = aggregate_metrics(all_metrics_for_aggregation)
        
        return global_metrics, all_page_scores, document_scores

    def score(self, pred_path, gt_path):
        """
        Score predictions. 
        Auto-detects flat/hierarchical structure.
        """
        wandb_run = self._init_wandb()
        
        structure_info = discover_dataset_structure(gt_path, self._get_score_file_extensions())
        
        if structure_info['type'] == 'empty':
            raise ValueError(f"No files found in {gt_path}")
        
        print(f"\n📊 Scoring: {structure_info['type']} structure")
        
        if structure_info['type'] == 'flat':
            # Find matching files
            gt_files = structure_info['images']
            pred_files = []
            matched_gt = []
            for gt_file in gt_files:
                pred_file = Path(pred_path) / Path(gt_file).name
                if pred_file.exists():
                    pred_files.append(str(pred_file))
                    matched_gt.append(gt_file)
            
            metrics_dict, page_scores = self._score_batch(pred_files, matched_gt, pred_path, gt_path)
            document_scores = None
            structure_type = 'flat'
        else:
            metrics_dict, page_scores, document_scores = self._score_hierarchical(pred_path, gt_path)
            structure_type = 'hierarchical'
        
        self._log_to_wandb(metrics_dict, wandb_run)
        self._display_metrics(metrics_dict)
        self._finish_wandb(wandb_run)
        
        # Return all data for CSV export
        return {
                'metrics': metrics_dict,
                'page_scores': page_scores,
                'document_scores': document_scores,
                'structure_type': structure_type
                }

    def _get_score_file_extensions(self):
        """
        File extensions for scoring. Default: same as prediction.
        """
        return self._get_file_extensions()
    

    def _filter_already_processed(self, file_paths, output_dir):
        """
        Filter files already predicted.
        """
        to_process = []
        for file_path in file_paths:
            output_name = Path(file_path).stem + '.xml'
            output_path = Path(output_dir) / output_name
            if not output_path.exists():
                to_process.append(file_path)
        return to_process
       
    def predict(self, data_path, output_dir, save_image=True, **kwargs):
        """
        Perform prediction on a dataset.
        
        Automatically detects dataset structure (flat or hierarchical) and
        processes accordingly. Preserves directory structure in output.
        
        Args:
            data_path: Path to dataset (can be flat or hierarchical)
            output_dir: Directory to save results
            save_image: Whether to copy source files to output
            **kwargs: Additional task-specific arguments
            
        Returns:
            List of results from processing
        """
        # Ensure model is loaded
        if not self.model:
            self.load()
        
        # Discover dataset structure
        file_extensions = self._get_file_extensions()
        structure_info = discover_dataset_structure(data_path, file_extensions)
        
        if structure_info['type'] == 'empty':
            raise ValueError(f"No files found in {data_path}")
        
        # Display structure information
        self._display_structure_info(structure_info)
        
        # Process according to structure
        if structure_info['type'] == 'flat':
            file_paths = self._filter_already_processed(structure_info['images'], output_dir)
            if not file_paths:
                print("✓ All files already processed")
                return []
            if len(structure_info['images']) - len(file_paths) > 0:
                print(f"  Skipping {len(structure_info['images']) - len(file_paths)} already processed files")

            return self._process_batch(
                file_paths=file_paths,
                source_dir=data_path,
                output_dir=output_dir,
                save_image=save_image,
                **kwargs
            )
        
        else:
            # Hierarchical structure: process by subdirectory
            return self._process_hierarchical(
                structure_info=structure_info,
                output_dir=output_dir,
                save_image=save_image,
                **kwargs
            )
    
    def _display_structure_info(self, structure_info):
        """Display information about the dataset structure."""
        print(f"\n📊 Dataset structure: {structure_info['type']}")
        print(f"📷 Total files: {len(structure_info['images'])}")
        
        if structure_info['type'] == 'hierarchical':
            print(f"📁 Subdirectories: {len(structure_info['subdirs'])}")
            print("\n⚙️  Processing hierarchical structure (preserving folders)...")
    
    def _process_hierarchical(self, structure_info, output_dir, save_image=True, **kwargs):
        """
        Process a hierarchical dataset structure.
        
        Args:
            structure_info: Structure information from discover_dataset_structure
            output_dir: Base output directory
            save_image: Whether to copy source files
            **kwargs: Additional arguments for _process_batch
            
        Returns:
            List of all results
        """
        all_results = []
        total_subdirs = len(structure_info['subdirs'])
        
        for idx, (subdir_path, files) in enumerate(structure_info['structure'].items(), start=1):
            # Create corresponding output subdirectory
            subdir_name = Path(subdir_path).name
            subdir_output = Path(output_dir) / subdir_name
            subdir_output.mkdir(parents=True, exist_ok=True)

            files_to_process = self._filter_already_processed(files, str(subdir_output))
            
            if not files_to_process:
                print(f"[{idx:02d}/{total_subdirs:02d}] ✓ {subdir_name}: All files already processed")
                continue
            
            skipped = len(files) - len(files_to_process)
            if skipped > 0:
                print(f"[{idx:02d}/{total_subdirs:02d}] 📁 {subdir_name}: Processing {len(files_to_process)}/{len(files)} files (skipping {skipped})")
            else:
                print(f"[{idx:02d}/{total_subdirs:02d}] 📁 {subdir_name}: Processing {len(files_to_process)} files")
            
            # Process this subdirectory
            try:
                results = self._process_batch(
                    file_paths=files_to_process,
                    source_dir=subdir_path,
                    output_dir=str(subdir_output),
                    save_image=save_image,
                    **kwargs
                )
                if results:
                    all_results.extend(results if isinstance(results, list) else [results])
            except Exception as e:
                print(f"  ❌ Error processing {subdir_name}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n✓ Processed {total_subdirs} subdirectories")
        return all_results
    
    def visualize(self, task_name, data_path, xml_path=None, output_dir=None):
        """
        Visualisation tool from xml object.
        
        Args:
            task_name: Name of the task for visualization type
            data_path: Path to images
            xml_path: path to xml files (if different from data_path)
            output_dir: Path where to save visualisations
            
        Returns:
            Number of successful visualizations
        """
        print(f"Visualizing results in {data_path}...")
        
        # Use xml_path if provided, otherwise same as data_path
        if xml_path is None:
            xml_path = data_path
        
        # Discover dataset structure
        file_extensions = self._get_file_extensions()
        structure_info = discover_dataset_structure(data_path, file_extensions)
        
        if structure_info['type'] == 'empty':
            print("No files found for visualization")
            return 0
        
        # Process according to structure
        if structure_info['type'] == 'flat':
            # Flat structure: use existing function
            return visualize_folder(
                img_dir=data_path,
                xml_dir=xml_path,
                output_dir=output_dir,
                visualization_type=task_name
            )
        else:
            # Hierarchical structure: process each subdirectory
            print(f"📁 Processing {len(structure_info['subdirs'])} subdirectories...")
            
            total_success = 0
            total_subdirs = len(structure_info['subdirs'])
            
            for idx, subdir_path in enumerate(structure_info['subdirs'], start=1):
                subdir_name = Path(subdir_path).name
                
                # Determine XML path for this subdirectory
                if xml_path != data_path:
                    subdir_xml = Path(xml_path) / subdir_name
                else:
                    subdir_xml = subdir_path
                
                # Determine output path for this subdirectory
                if output_dir:
                    subdir_output = Path(output_dir) / subdir_name
                    subdir_output.mkdir(parents=True, exist_ok=True)
                else:
                    subdir_output = None
                
                try:
                    print(f"[{idx:02d}/{total_subdirs:02d}] 📁 {subdir_name}:")
                    
                    success_count = visualize_folder(
                        img_dir=str(subdir_path),
                        xml_dir=str(subdir_xml),
                        output_dir=str(subdir_output) if subdir_output else None,
                        visualization_type=task_name
                    )
                    
                    total_success += success_count
                    print(f"  ✓ Generated {success_count} visualizations")
                    
                except Exception as e:
                    print(f"  ❌ Error: {e}")
            
            print(f"\n✓ Total: {total_success} visualizations across {total_subdirs} subdirectories")
            return total_success
    
    def _get_file_extensions(self):
        """
        Get the file extensions to look for.
        
        Override this in subclasses if needed.
        
        Returns:
            List of file extensions (e.g., ['*.jpg', '*.png'])
        """
        return ['*.jpg', '*.jpeg', '*.png']
    
    def _should_save_source_files(self):
        """
        Whether to copy source files to output directory.
        
        Override in subclasses if different behavior needed.
        
        Returns:
            Boolean
        """
        return True
    
    def _get_progress_description(self):
        """
        Get the description for progress bar.
        
        Returns:
            String for tqdm description
        """
        return "Processing"