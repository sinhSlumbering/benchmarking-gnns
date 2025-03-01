"""
Simple training scheme for TSP edge classification using existing training functions
"""
import os
import time
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# Import necessary modules
from nets.TSP_edge_classification.load_net import gnn_model
from data.load_data import LoadData
# Import existing training functions
from train.train_TSP_edge_classification import train_epoch_sparse, evaluate_network_sparse


def setup_device(use_gpu, gpu_id):
    """Set up and return the device."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def load_config(config_path):
    """Load and return the configuration."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def setup_directories(out_dir, model_name, dataset_name, gpu_id):
    """Create and return directory paths for logs, checkpoints, and results."""
    timestamp = time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    
    # Create directory names with timestamp
    log_dir = os.path.join(out_dir, 'logs', f"{model_name}_{dataset_name}_GPU{gpu_id}_{timestamp}")
    ckpt_dir = os.path.join(out_dir, 'checkpoints', f"{model_name}_{dataset_name}_GPU{gpu_id}_{timestamp}")
    result_file = os.path.join(out_dir, 'results', f"result_{model_name}_{dataset_name}_GPU{gpu_id}_{timestamp}.txt")
    config_file = os.path.join(out_dir, 'configs', f"config_{model_name}_{dataset_name}_GPU{gpu_id}_{timestamp}.txt")
    
    # Create directories if they don't exist
    os.makedirs(os.path.join(out_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'configs'), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    return log_dir, ckpt_dir, result_file, config_file


def count_parameters(model):
    """Count and return the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def train(model, dataset, device, params, net_params, dirs):
    """Main training function using existing train_epoch_sparse and evaluate_network_sparse functions."""
    log_dir, ckpt_dir, result_file, config_file = dirs
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # Save configuration
    with open(config_file, 'w') as f:
        f.write(f"Dataset: {dataset.name}\n")
        f.write(f"Model: {net_params['model']}\n\n")
        f.write(f"params={str(params)}\n\n")
        f.write(f"net_params={str(net_params)}\n\n")
        f.write(f"Total Parameters: {count_parameters(model)}\n")
    
    # Create dataloaders
    train_loader = DataLoader(dataset.train, batch_size=params['batch_size'], shuffle=True, 
                             collate_fn=dataset.collate, num_workers=params.get('num_workers', 4))
    val_loader = DataLoader(dataset.val, batch_size=params['batch_size'], shuffle=False, 
                           collate_fn=dataset.collate, num_workers=params.get('num_workers', 4))
    test_loader = DataLoader(dataset.test, batch_size=params['batch_size'], shuffle=False, 
                            collate_fn=dataset.collate)
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=params['lr_reduce_factor'],
        patience=params['lr_schedule_patience'], verbose=True
    )
    
    # Track metrics
    best_val_f1 = 0
    best_epoch = 0
    early_stopping_counter = 0
    early_stopping_patience = params.get('early_stopping_patience', 20)
    early_stopping = params.get('early_stopping', True)
    
    # Timing variables
    t0 = time.time()
    per_epoch_time = []
    
    # Training loop
    print("Starting training...")
    for epoch in range(params['epochs']):
        epoch_start_time = time.time()
        
        # Train using existing train_epoch_sparse function
        epoch_train_loss, epoch_train_f1, optimizer, train_metrics = train_epoch_sparse(
            model, optimizer, device, train_loader, epoch)
        
        # Validate using existing evaluate_network_sparse function
        epoch_val_loss, epoch_val_f1, val_metrics = evaluate_network_sparse(
            model, device, val_loader, epoch)
        
        # Update learning rate
        scheduler.step(epoch_val_loss)
        
        # Early stopping logic
        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            best_epoch = epoch
            early_stopping_counter = 0
            
            # Save best model
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_model.pt'))
        else:
            early_stopping_counter += 1
        
        # Track epoch time
        epoch_time = time.time() - epoch_start_time
        per_epoch_time.append(epoch_time)
        
        # Log results
        writer.add_scalar('train/loss', epoch_train_loss, epoch)
        writer.add_scalar('val/loss', epoch_val_loss, epoch)
        writer.add_scalar('train/f1', epoch_train_f1, epoch)
        writer.add_scalar('val/f1', epoch_val_f1, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{params['epochs']} | Time: {epoch_time:.2f}s | "
              f"Train Loss: {epoch_train_loss:.4f} | Train F1: {epoch_train_f1:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | Val F1: {epoch_val_f1:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save model
        if (epoch + 1) % params.get('save_epoch_interval', 10) == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f'epoch_{epoch+1}.pt'))
        
        # Check for early stopping
        if early_stopping and early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Check for learning rate
        if optimizer.param_groups[0]['lr'] < params['min_lr']:
            print("Learning rate below minimum threshold, stopping training")
            break
        
        # Check for maximum training time
        if (time.time() - t0) > params['max_time'] * 3600:
            print(f"Maximum training time reached ({params['max_time']} hours)")
            break
    
    # Load best model for testing
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'best_model.pt')))
    
    # Test the model using existing evaluate_network_sparse function
    test_loss, test_f1, test_metrics = evaluate_network_sparse(model, device, test_loader, epoch)
    print(f"Test Results: Loss: {test_loss:.4f} | F1: {test_f1:.4f}")
    
    # Write final results
    total_time = time.time() - t0
    avg_epoch_time = np.mean(per_epoch_time)
    with open(result_file, 'w') as f:
        f.write(f"Dataset: {dataset.name}\n")
        f.write(f"Model: {net_params['model']}\n")
        f.write(f"Total Parameters: {count_parameters(model)}\n\n")
        f.write("FINAL RESULTS\n")
        f.write(f"Best Epoch: {best_epoch+1}\n")
        f.write(f"Best Val F1: {best_val_f1:.4f}\n")
        f.write(f"Test F1: {test_f1:.4f}\n\n")
        f.write(f"Total Time: {total_time/3600:.2f} hours\n")
        f.write(f"Avg Epoch Time: {avg_epoch_time:.2f} seconds\n")
    
    writer.close()
    
    return test_f1


def main():
    parser = argparse.ArgumentParser(description='Simple TSP Edge Classification')
    parser.add_argument('--config', default='configs/TSP_edge_classification_GatedGCN_100k.json', 
                        help='Path to config file')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--model', help='Model name')
    parser.add_argument('--dataset', help='Dataset name')
    parser.add_argument('--out_dir', default='./out/', help='Output directory')
    parser.add_argument('--seed', type=int, help='Random seed')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = setup_device(config['gpu']['use'], args.gpu_id if args.gpu_id is not None else config['gpu']['id'])
    
    # Set model and dataset names (from args or config)
    model_name = args.model if args.model else config['model']
    dataset_name = args.dataset if args.dataset else config['dataset']
    
    # Create output directories
    out_dir = args.out_dir if args.out_dir else config.get('out_dir', './out/')
    dirs = setup_directories(out_dir, model_name, dataset_name, args.gpu_id if args.gpu_id is not None else config['gpu']['id'])
    
    # Load dataset
    dataset = LoadData(dataset_name)
    
    # Set parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = args.seed
    
    # Set network parameters
    net_params = config['net_params']
    net_params['model'] = model_name
    net_params['device'] = device
    net_params['in_dim'] = dataset.train[0][0].ndata['feat'][0].shape[0]
    net_params['in_dim_edge'] = dataset.train[0][0].edata['feat'][0].size(0)
    net_params['n_classes'] = len(np.unique(np.concatenate(dataset.train[:][1])))
    
    # Set seed for reproducibility
    set_seed(params['seed'])
    
    # Initialize model
    model = gnn_model(model_name, net_params)
    model = model.to(device)
    
    # Count parameters
    num_params = count_parameters(model)
    print(f"Total number of parameters: {num_params}")
    
    # Train model
    test_f1 = train(model, dataset, device, params, net_params, dirs)
    
    print(f"Training completed. Final Test F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()
