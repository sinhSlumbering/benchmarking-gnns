import os
import sys
import torch
import pickle
import time
import logging
import psutil
import argparse
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tsp_dataset import TSP  # Import TSP from tsp_dataset

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Prepare TSP dataset.')
    parser.add_argument('--graph_size', type=int,
                        required=True, help='Size of the TSP graph')
    args = parser.parse_args()

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
    # Points to 'data/' directory
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_pkl = os.path.join(data_dir, f'{DATASET_NAME}.pkl')

    splits = ['train', 'val', 'test']
    datasets = []

    for split in splits:
        logging.info(f"Processing {split} dataset...")
        split_start_time = time.time()
        dataset = TSP(data_dir=data_dir, split=split, graph_size=args.graph_size, num_neighbors=args.graph_size - 1,
                      max_samples=100000, num_workers=None)  # Adjust num_workers as needed

        logging.info(f"Finished processing {split} dataset in {
                     time.time() - split_start_time:.2f} seconds.")

        # Log memory usage
        mem_info = process.memory_info()
        logging.info(f"Memory Usage after processing {split}: {
                     mem_info.rss / (1024 ** 2):.2f} MB")

        # Save each split separately
        split_output_pkl = os.path.join(
            data_dir, f'{DATASET_NAME}_{split}.pkl')
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

    logging.info(f"Data processing completed in {
                 time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
