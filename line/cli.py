from core.cli import BaseCLI
from core.config_manager import ConfigManager
from line.model import LineModel
import argparse
import os
import sys
import pandas as pd

class LineCLI(BaseCLI):
    """
    CLI interface for line segmentation.
    """
    
    def __init__(self):
        """
        Initialize the line segmentation CLI interface.
        """
        super().__init__("Line segmentation evaluation and training tool")
        
    def _add_specific_arguments(self):
        """
        Add line-specific command line arguments.
        """
        self.parser.add_argument('--text_direction', default='horizontal-lr', choices=['horizontal-lr', 'vertical-rl'],
                            help="Text direction for line segmentation")
        self.parser.add_argument('--iou_threshold', type=float, default=None,
                            help="IoU threshold for evaluation")
    
    def run(self, args=None):
        """
        Run the line CLI interface.
        
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
            default_config = config_manager.get_default_config('line')
            configs.append(default_config)
        
        # Update configurations with CLI arguments
        for config in configs:
            
            # Update other parameters
            if parsed_args.text_direction:
                config["text_direction"] = parsed_args.text_direction
                
            if parsed_args.iou_threshold:
                config["iou_threshold"] = parsed_args.iou_threshold
        
        # Validate configurations
        self._validate_configs()
        
        # Create models from configurations
        models = [LineModel(config) for config in configs]
        
        # Store results for all models
        all_results = {}
        
        if parsed_args.function == 'eval':
            for model in models:
                print(f"\nEvaluating model: {model.name}")
                
                results = model.evaluate(corpus_path=parsed_args.corpus_path)
                all_results[model.name] = results
                
        elif parsed_args.function == 'train':
            for model in models:
                print(f"\nTraining and evaluating model: {model.name}")

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
    