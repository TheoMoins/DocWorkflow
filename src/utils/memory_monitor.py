import os
import psutil
import gc


def get_fast_vision_model():
    try:
        from unsloth import FastVisionModel
        return FastVisionModel
    except (ImportError, NotImplementedError):
        return None

class MemoryLimitExceeded(RuntimeError):
    """Raised to stop processing before system memory runs out."""


def get_memory_usage_percent():
    """Get current system memory usage as percentage."""
    return psutil.virtual_memory().percent


def get_available_memory_gb():
    """Get available system memory in GB."""
    return psutil.virtual_memory().available / (1024 ** 3)


def check_memory_safe(min_available_gb=2.0, max_usage_percent=90):
    """
    Check if it's safe to continue processing based on memory.
    
    Args:
        min_available_gb: Minimum available memory in GB
        max_usage_percent: Maximum memory usage percentage
        
    Returns:
        Tuple (is_safe, message)
    """
    available_gb = get_available_memory_gb()
    usage_percent = get_memory_usage_percent()
    
    if available_gb < min_available_gb:
        return False, f"Low memory: {available_gb:.1f}GB available (min: {min_available_gb}GB)"
    
    if usage_percent > max_usage_percent:
        return False, f"High memory usage: {usage_percent:.1f}% (max: {max_usage_percent}%)"
    
    return True, None


def force_garbage_collection():
    """Force garbage collection and return memory freed."""
    gc.collect()
    
    # If using PyTorch, also clear CUDA cache
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def ensure_memory_safe(min_available_gb=2.0, max_usage_percent=90):
    """
    Check memory, collect garbage once if needed, and abort if still unsafe.

    Raises:
        MemoryLimitExceeded: if memory stays above limits after garbage collection
    """
    is_safe, message = check_memory_safe(min_available_gb, max_usage_percent)
    if is_safe:
        return

    force_garbage_collection()
    is_safe, message = check_memory_safe(min_available_gb, max_usage_percent)
    if not is_safe:
        raise MemoryLimitExceeded(message)