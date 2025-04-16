import argparse
from abc import ABC, abstractmethod
import os
import pandas as pd

from core.config_manager import ConfigManager

from layout.model import LayoutModel
from line.model import LineModel

class CLI:
    """
    Base class for command line interfaces.
    """
    
    def __init__(self, module):
        """
        Initialize the CLI interface.
        
        Args:
            module: "line" or "layout"
        """
        self.parser = argparse.ArgumentParser()
        self._add_common_arguments()
        self._add_specific_arguments()
        self.module = module
        
    def _add_common_arguments(self):
        """
        Add common command line arguments.
        """
        self.parser.add_argument('--function', choices=['eval', 'train', 'predict'], required=True,
                           help="Function to run: 'eval' for evaluation, 'train' for training")
        self.parser.add_argument('--data_path', default=None,
                           help="Path to training/validation/test data config file (overrides config file)")
        self.parser.add_argument('--corpus_path', default=None,
                           help="Optional path to corpus data for additional testing (overrides config file")
        self.parser.add_argument('--pred_path', default=None,
                           help="Path to prediction files (only for prediction)")
        self.parser.add_argument('--configs', help='Path to JSON configuration file(s)', nargs='*',
                           required=False)
        self.parser.add_argument('--output', default=None,
                           help="Optional path to save evaluation results as CSV")

    def _get_model_class(self):
        """
        Return the model class to use for this CLI.
        
        Returns:
            Class: The model class to instantiate
        """
        if self.module == 'line':
            return LineModel
        elif self.module == 'layout':
            return LayoutModel
        else: 
            raise ValueError('Unknown module pass to the run function.')


    def _validate_configs(self):
        """
        TODO : Validate configuration.
        
        Args:
            configs: List of configuration dictionaries
            function: Function to be performed ('eval' or 'train')
        """

        return
        # if self.function == 'eval' and not self.configs["data_path"]:
        #     print(f"Warning: No test path specified for model {self.configs['name']}")
            
        # if self.function == 'train' and not self.configs["data_path"]:
        #     print(f"Warning: No training data path specified for model {self.configs['name']}")


    @abstractmethod
    def _add_specific_arguments(self):
        """Add component-specific command line arguments."""
        pass
    
    def _update_config(self, config, args):
        """
        Update config with command line arguments.
        
        Args:
            config: Configuration dictionary
            args: Parsed command line arguments
            
        Returns:
            dict: Updated configuration
        """
        # Override data_path if provided in CLI
        if args.data_path:
            config['data_path'] = args.data_path
        
        # Override corpus_path if provided in CLI
        if args.corpus_path:
            config['corpus_path'] = args.corpus_path

        return config

    def run(self, args=None):
        """
        Run the CLI interface with arguments.
        
        Args:
            args: Optional list of command line arguments
        """
        parsed_args = self.parser.parse_args(args)
        
        # Load configurations
        config_manager = ConfigManager()
        configs = []
        
        if parsed_args.configs:
            configs = config_manager.load_configs(parsed_args.configs)
        else:
            # Use default configuration if no model is specified
            default_config = config_manager.get_default_config(self.module)
            configs.append(default_config)
        
        # Update configurations with CLI arguments
        for config in configs:
            self._update_config(config, parsed_args)
        
        # Validate configurations
        self._validate_configs()
        
        # Create models from configurations
        model_class = self._get_model_class()
        models = [model_class(config) for config in configs]
        
        # Store results for all models
        all_results = {}
        
        if parsed_args.function == 'eval':
            for model in models:
                print(f"\nEvaluating model: {model.name}")
                
                results = model.evaluate(corpus_path=model.config.get("corpus_path"))
                all_results[model.name] = results
                
        elif parsed_args.function == 'train':
            for model in models:
                print(f"\nTraining and evaluating model: {model.name}")

                model.train()                
                results = model.evaluate(corpus_path=model.config.get("corpus_path"))
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


    def save_results(self, results, output_path):
        """
        Save evaluation results to a CSV file.
        
        Args:
            results: DataFrame with evaluation results
            output_path: Path to save the CSV file
        """
        results.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")