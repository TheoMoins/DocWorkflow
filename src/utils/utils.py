import torch
import wandb

import glob
import os


def setup_wandb(key=None):
    """
    Configure Weights & Biases for logging.
    
    Args:
        key: Optional WandB API key
    """
    if key:
        wandb.login(key=key)
    else:
        wandb.login()

def get_device():
    """
    Determine the device to use (CUDA or CPU).
    
    Returns:
        String indicating the device ('cuda' or 'cpu')
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def find_files(directory, pattern):
    """
    Find all files matching a pattern in a directory.
    
    Args:
        directory: Directory to search
        pattern: File pattern (e.g., "*.xml", "*.jpg")
        
    Returns:
        List of matching file paths
    """
    return sorted(glob.glob(os.path.join(directory, pattern)))

def ensure_dir(directory):
    """
    Ensure a directory exists, create it if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        Directory path
    """
    os.makedirs(directory, exist_ok=True)
    return directory