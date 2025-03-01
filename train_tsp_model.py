#!/usr/bin/env python

"""
Script to train a GNN model on the TSP edge classification task with optimal configuration
"""

import argparse
import json
import os
import time
from main_TSP_edge_classification import train_val_pipeline, train_val_pipeline_chunked, gpu_setup, view_model_param
from data.load_data import LoadData
from data.stream_data import StreamingTSPDataset

def main():
    """
    Main function to set up and train a model for TSP edge classification
    """
    
    # Default optimal configuration
    config = {
        "gpu": {
            "use": True,
            "id": 0
        },
        "model": "GatedGCN",  # GatedGCN has shown good performance on graph tasks
        "dataset": "TSP",
        "out_dir": "./output/",
        "params": {
            "seed": 41,
            "epochs": 1000,
            "batch_size": 64,
            "init_lr": 0.001,
            "lr_reduce_factor": 0.5,
            "lr_schedule_patience": 10,
            "min_lr": 1e-6,
            "weight_decay": 0.0,
            "print_epoch_interval": 5,
            "max_time": 24,
            "use_amp": True,  # Use Automatic Mixed Precision for faster training
            "use_chunked_training": True,  # Use chunked training for large datasets
            "num_workers": 4,  # Number of dataloader workers
            "pin_memory": True  # Pin memory for faster data transfer
        },
        "net_params": {
            "L": 4,  # Number of GNN layers
            "hidden_dim": 128,
            "out_dim": 128,
            "residual": True,
            "edge_feat": True,
            "readout": "mean",
            "in_feat_dropout": 0.0,
            "dropout": 0.0,
            "batch_norm": True,
            "layer_norm": False,
            "layer_type": "edgereprfeat",
            "self_loop": True
        }
    }
    
    # Parse command line arguments to override config
    parser = argparse.ArgumentParser(description='Train a GNN model for TSP edge classification')
    parser.add_argument('--config', type=str, help='Path to config file (will override default config)')
    parser.add_argument('--gpu_id', type=int, help='GPU ID to use')
    parser.add_argument('--model', type=str, help='Model to use (e.g., GatedGCN, GCN, GAT)')
    parser.add_argument('--dataset', type=str, help='Dataset to use')
    parser.add_argument('--streaming', action='store_true', help='Use streaming dataset (for large datasets)')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    args = parser.parse_args()
    
    # If config file is provided, load it
    if args.config:
        with open(args.config) as f:
            custom_config = json.load(f)
            # Update config with custom config
            for k, v in custom_config.items():
                if isinstance(v, dict) and k in config:
                    config[k].update(v)
                else:
                    config[k] = v
    
    # Override config with command line args
    if args.gpu_id is not None:
        config['gpu']['id'] = args.gpu_id
    if args.model:
        config['model'] = args.model
    if args.dataset:
        config['dataset'] = args.dataset
    if args.batch_size:
        config['params']['batch_size'] = args.batch_size
    if args.epochs:
        config['params']['epochs'] = args.epochs
    
    # Ensure output directories exist
    os.makedirs(config['out_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['out_dir'], 'logs'), exist_ok=True)
    os.makedirs(os.path.join(config['out_dir'], 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(config['out_dir'], 'results'), exist_ok=True)
    os.makedirs(os.path.join(config['out_dir'], 'configs'), exist_ok=True)
    
    # Set up GPU
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    config['net_params']['device'] = device
    config['net_params']['gpu_id'] = config['gpu']['id']
    config['net_params']['batch_size'] = config['params']['batch_size']
    
    # Load dataset
    if args.streaming:
        print(f"Loading streaming dataset {config['dataset']}...")
        dataset = StreamingTSPDataset(config['dataset'])
    else:
        print(f"Loading dataset {config['dataset']}...")
        dataset = LoadData(config['dataset'])
    
    # Set input dimensions based on dataset
    config['net_params']['in_dim'] = dataset.train[0][0].ndata['feat'][0].shape[0]
    config['net_params']['in_dim_edge'] = dataset.train[0][0].edata['feat'][0].size(0)
    
    # Calculate number of classes from the dataset
    import numpy as np
    num_classes = len(np.unique(np.concatenate(dataset.train[:][1])))
    config['net_params']['n_classes'] = num_classes
    
    print(f"Using model: {config['model']}")
    print(f"Training for {config['params']['epochs']} epochs with batch size {config['params']['batch_size']}")
    print(f"Number of classes: {num_classes}")
    
    # Set up directories for logs, checkpoints, and results
    timestamp = time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    model_name = config['model']
    dataset_name = config['dataset']
    gpu_id = str(config['gpu']['id'])
    
    log_dir = os.path.join(config['out_dir'], 'logs', f"{model_name}_{dataset_name}_GPU{gpu_id}_{timestamp}")
    ckpt_dir = os.path.join(config['out_dir'], 'checkpoints', f"{model_name}_{dataset_name}_GPU{gpu_id}_{timestamp}")
    result_file = os.path.join(config['out_dir'], 'results', f"result_{model_name}_{dataset_name}_GPU{gpu_id}_{timestamp}")
    config_file = os.path.join(config['out_dir'], 'configs', f"config_{model_name}_{dataset_name}_GPU{gpu_id}_{timestamp}")
    
    dirs = (log_dir, ckpt_dir, result_file, config_file)
    
    # Calculate total parameters
    config['net_params']['total_param'] = view_model_param(config['model'], config['net_params'])
    
    print(f"Model has {config['net_params']['total_param']} parameters")
    
    # Train model using chunked or standard pipeline
    # if config['params'].get('use_chunked_training', False):
    print("Using chunked training pipeline...")
    train_val_pipeline_chunked(config['model'], dataset, config['params'], config['net_params'], dirs)
    # else:
    #     print("Using standard training pipeline...")
    #     train_val_pipeline(config['model'], dataset, config['params'], config['net_params'], dirs)

if __name__ == "__main__":
    main()
