import os
import pickle
import argparse
import numpy as np
import dgl
from tqdm import tqdm

def prepare_dataset_for_streaming(source_file, output_dir, split_ratio=[0.8, 0.1, 0.1]):
    """
    Prepares a dataset for streaming by splitting it into individual files.
    
    Args:
        source_file (str): Path to the source dataset file (typically a single large file)
        output_dir (str): Directory to save the individual graph files
        split_ratio (list): Train/val/test split ratios
    """
    print(f"Loading dataset from {source_file}...")
    with open(source_file, 'rb') as f:
        dataset = pickle.load(f)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    
    # Shuffle dataset
    np.random.seed(42)
    indices = np.random.permutation(len(dataset))
    
    # Calculate split sizes
    train_size = int(len(dataset) * split_ratio[0])
    val_size = int(len(dataset) * split_ratio[1])
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Save each graph to an individual file
    print("Processing training set...")
    for i, idx in enumerate(tqdm(train_indices)):
        graph = dataset[idx]
        with open(os.path.join(output_dir, 'train', f'graph_{i:06d}.pkl'), 'wb') as f:
            pickle.dump(graph, f)
    
    print("Processing validation set...")
    for i, idx in enumerate(tqdm(val_indices)):
        graph = dataset[idx]
        with open(os.path.join(output_dir, 'val', f'graph_{i:06d}.pkl'), 'wb') as f:
            pickle.dump(graph, f)
    
    print("Processing test set...")
    for i, idx in enumerate(tqdm(test_indices)):
        graph = dataset[idx]
        with open(os.path.join(output_dir, 'test', f'graph_{i:06d}.pkl'), 'wb') as f:
            pickle.dump(graph, f)
    
    print(f"Dataset prepared and saved to {output_dir}")
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare dataset for streaming')
    parser.add_argument('--source', type=str, required=True, help='Source dataset file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation set ratio')
    
    args = parser.parse_args()
    
    # Calculate test ratio
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio <= 0:
        raise ValueError("Train and val ratios sum to >= 1")
    
    prepare_dataset_for_streaming(
        args.source, 
        args.output, 
        split_ratio=[args.train_ratio, args.val_ratio, test_ratio]
    )
