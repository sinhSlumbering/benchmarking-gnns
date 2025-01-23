import os
import sys
import numpy as np
import torch
import pickle
import time
import multiprocessing
import logging
import psutil

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tsp_dataset import TSP  # Import TSP from tsp_dataset

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting data preparation script.")

    # Log initial memory usage
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logging.info(f"Initial Memory Usage: {mem_info.rss / (1024 ** 2):.2f} MB")

    start_time = time.time()
    
    DATASET_NAME = 'TSP'
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Points to 'data/' directory
    output_pkl = os.path.join(data_dir, f'{DATASET_NAME}.pkl')
    
    splits = ['train', 'val', 'test']
    datasets = []
    
    for split in splits:
        logging.info(f"Processing {split} dataset...")
        split_start_time = time.time()
        dataset = TSP(data_dir=data_dir, split=split, num_neighbors=99,
                      max_samples=100000, num_workers=None)  # Adjust num_workers as needed

        logging.info(f"Finished processing {split} dataset in {time.time() - split_start_time:.2f} seconds.")
        
        # Log memory usage
        mem_info = process.memory_info()
        logging.info(f"Memory Usage after processing {split}: {mem_info.rss / (1024 ** 2):.2f} MB")
        
        # Save each split separately
        split_output_pkl = os.path.join(data_dir, f'{DATASET_NAME}_{split}.pkl')
        with open(split_output_pkl, 'wb') as f:
            pickle.dump((dataset.graph_lists, dataset.edge_labels), f)
        logging.info(f"Saved {split} dataset to {split_output_pkl}.")
        
        datasets.append((dataset.graph_lists, dataset.edge_labels))
        
        # Clean up to free memory
        del dataset
        torch.cuda.empty_cache()
    
    # Combine all splits into one pickle file
    logging.info(f"Combining datasets into {output_pkl}...")
    with open(output_pkl, 'wb') as f:
        pickle.dump(datasets, f)
    logging.info("Combined all datasets and saved.")
    
    logging.info(f"Data processing completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()