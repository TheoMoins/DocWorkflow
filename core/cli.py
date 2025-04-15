import argparse
from abc import ABC, abstractmethod
import os
import pandas as pd

class BaseCLI(ABC):
    """
    Base class for command line interfaces.
    """
    
    def __init__(self, description):
        """
        Initialize the CLI interface.
        
        Args:
            description: CLI interface description
        """
        self.parser = argparse.ArgumentParser(description=description)
        self._add_common_arguments()
        self._add_specific_arguments()
        
    def _add_common_arguments(self):
        """
        Add common command line arguments.
        """
        self.parser.add_argument('--function', choices=['eval', 'train'], required=True,
                           help="Function to run: 'eval' for evaluation, 'train' for training")
        self.parser.add_argument('--data_path', default=None,
                           help="Path to training/validation/test data config file (overrides config file)")
        self.parser.add_argument('--corpus_path', default=None,
                           help="Optional path to corpus data for additional testing (overrides config file")
        self.parser.add_argument('--pred_path', default=None,
                           help="Path to prediction files (only for prediction)")
        self.parser.add_argument('--config', help='Path to JSON configuration file(s)', nargs='*',
                           required=False)
        self.parser.add_argument('--output', default=None,
                           help="Optional path to save evaluation results as CSV")

    def _validate_configs(self):
        """
        TODO : Validate configuration.
        
        Args:
            configs: List of configuration dictionaries
            function: Function to be performed ('eval' or 'train')
        """

        return
        # if self.function == 'eval' and not self.config["data_path"]:
        #     print(f"Warning: No test path specified for model {self.config['name']}")
            
        # if self.function == 'train' and not self.config["data_path"]:
        #     print(f"Warning: No training data path specified for model {self.config['name']}")


    @abstractmethod
    def _add_specific_arguments(self):
        """Add component-specific command line arguments."""
        pass
    
    @abstractmethod
    def run(self, args=None):
        """
        Run the CLI interface with arguments.
        
        Args:
            args: Optional list of command line arguments
        """
        pass
    
    def save_results(self, results, output_path):
        """
        Save evaluation results to a CSV file.
        
        Args:
            results: DataFrame with evaluation results
            output_path: Path to save the CSV file
        """
        results.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")