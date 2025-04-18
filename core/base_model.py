from abc import ABC, abstractmethod
import torch
import os
import wandb
from datetime import datetime
import pandas as pd
import tabulate

from core.visualisation import visualize_folder

class BaseModel(ABC):
    """
    Abstract base class for all document analysis models.
    """
    
    def __init__(self, config):
        """
        Initialize the model with its configuration.
        
        Args:
            config: Model configuration dictionary
            models_dir: Directory containing model weights
        """
        self.config = config
        self.name = config.get('name', 'unknown_model')
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
        
        return wandb.init(
            project=self.wandb_project, 
            name=f"eval-{self.name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
    
    @abstractmethod
    def _compute_metrics(self, data_path):
        """
        Compute raw evaluation metrics for the model.

        Returns:
            Dictionary of raw evaluation metrics
        """
        pass
    
    def _display_metrics(self, metrics):
        """
        Display evaluation metrics in a formatted way.
        
        Args:
            metrics: Dictionary of evaluation metrics
        """
        # Create a formatted table for display
        table = [["Metric", "Value"]]
        
        # Add each metric to the table with proper formatting
        for key, value in metrics.items():
            # Format numeric values to 4 decimal places
            if isinstance(value, (int, float)):
                formatted_value = round(value, 4)
            else:
                formatted_value = value
            
            # Add to table
            table.append([key, formatted_value])
        
        # Display the table
        print("\nEvaluation Results:")
        print(tabulate.tabulate(table, tablefmt="grid", headers="firstrow"))
    

    @abstractmethod
    def predict(self, image_path, save=False):
        """
        Perform prediction on an image.
        
        Args:
            image_path: Path to the image
            save: boolean whether to save the image with the prediction or not
            
        Returns:
            Prediction results
        """
        pass

    @abstractmethod
    def train(self, train_path=None, **kwargs):
        """
        Train model using a given dataset
        
        Args:
            train_path: Path to a training/validation set.
        """
        pass
    
    def evaluate(self, corpus_path=None):
        """
        Evaluate model and handle logging and result presentation.
        
        Args:
            corpus_path: Optional path to additional corpus data
            log_to_wandb: Whether to log results to Weights & Biases
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Initialize wandb if needed
        run = self._init_wandb()
        
        # Compute metrics using model-specific implementation
        metrics = self._compute_metrics(self.config["data_path"])
        if corpus_path:
            corpus_metrics = self._compute_metrics(self.config["corpus_path"], is_corpus=True)
            metrics = {**metrics, **corpus_metrics}
        
        # Display metrics in a formatted way
        self._display_metrics(metrics)
        
        self._log_to_wandb(metrics, run)
        self._finish_wandb(run)
        
        return metrics
    
    def visualize(self, corpus_path, xml_path=None, output_dir=None):
        """
        Visualisation tool from xml object.
        
        Args:
            corpus_path: Path to images
            xml_path: path to xml files (if different from corpus_path)
            output_dir: Path where to save visualisations
            
        Returns:
            Nombre de visualisations réussies
        """
        print(f"Visualizing results in {corpus_path}...")
        
        # Déterminer le type de visualisation en fonction du type de modèle
        visualization_type = 'layout' if self.__class__.__name__ == 'LayoutModel' else 'line'
        
        # Utiliser l'utilitaire de visualisation
        return visualize_folder(
            img_dir=corpus_path,
            xml_dir=xml_path,
            output_dir=output_dir,
            visualization_type=visualization_type
        )
