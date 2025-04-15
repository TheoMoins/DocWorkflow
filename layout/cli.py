from core.cli import BaseCLI
from core.config_manager import ConfigManager
from layout.model import LayoutModel
import argparse
import os
import sys
import pandas as pd

class LayoutCLI(BaseCLI):
    """
    CLI interface for layout segmentation.
    """
    
    def __init__(self):
        """
        Initialize the layout segmentation CLI interface.
        """
        super().__init__("Layout segmentation evaluation and training tool")
        self.args = None

    def _add_specific_arguments(self):
        """
        Add line-specific command line arguments.
        """
        return

    def run(self, args=None):
        """
        Run the layout CLI interface.
        
        Args:
            args: Optional list of command line arguments
        """
        parsed_args = self.parser.parse_args(args)
        
        # Load configurations
        config_manager = ConfigManager()
        configs = []
        
        if parsed_args.config:
            configs = config_manager.load_config(parsed_args.config)

        else:
            # Use default configuration if no model is specified
            default_config = config_manager.get_default_config('layout')
            configs.append(default_config)
        
        # Validate configurations
        self._validate_configs()
                
        # Create models from configurations
        models = [LayoutModel(config) for config in configs]
        
        # Store results for all models
        all_results = {}
        
        if parsed_args.function == 'eval':
            for model in models:
                print(f"\nEvaluating model: {model.name}")
                    
                # Evaluate model
                results = model.evaluate(corpus_path=parsed_args.corpus_path)
                all_results[model.name] = results
                
        elif parsed_args.function == 'train':
            for model in models:
                print(f"\nTraining and evaluating model: {model.name}")

                # Train model
                model.train()                
                results = model.evaluate(corpus_path=parsed_args.corpus_path)
                all_results[model.name] = results
        
        # Convert results to DataFrame for multiple models
        if len(all_results) > 1:
            df_results = pd.DataFrame(all_results).T
            df_results.index.name = "model"
            df_results.reset_index(inplace=True)
            
            # Print comparison
            print("\nModel Comparison:")
            print(df_results.to_string())
            
            # Save results if output path is provided
            if parsed_args.output:
                self.save_results(df_results, parsed_args.output)
        elif len(all_results) == 1:
            # Save single model results if output path is provided
            if parsed_args.output:
                model_name = list(all_results.keys())[0]
                self.save_results(all_results[model_name], parsed_args.output)
    