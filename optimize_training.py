import torch
import gc
import os
import psutil
import time

def get_memory_usage():
    """Return the memory usage in MB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 ** 2  # Convert to MB

def print_memory_stats():
    """Print memory usage statistics"""
    print(f"CPU Memory: {get_memory_usage():.2f} MB")
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

def clear_memory():
    """Clear unused memory"""
    gc.collect()
    torch.cuda.empty_cache()
    
class Timer:
    """Simple timer for performance profiling"""
    def __init__(self, name=""):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        print(f"{self.name} took {elapsed:.4f} seconds")
