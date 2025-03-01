import torch
import torch.optim as optim
import torch.cuda.amp as amp
import numpy as np
import os
import time
import random
import glob
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter
from tqdm import tqdm

from nets.TSP_edge_classification.load_net import gnn_model
from train.train_TSP_edge_classification import evaluate_network_sparse as evaluate_network
from optimize_training import clear_memory, print_memory_stats, Timer

def count_edge_types(preds, labels):
    """Helper function to count correct predictions and totals for each edge type"""
    correct_counts = {}
    total_counts = {}
    false_positives = {}
    
    for i in range(len(labels)):
        label = int(labels[i])
        pred = int(preds[i])
        
        # Count total occurrences
        total_counts[label] = total_counts.get(label, 0) + 1
        
        # Count correct predictions
        if pred == label:
            correct_counts[label] = correct_counts.get(label, 0) + 1
            
        # Count false positives (predicted 1 when actual was 0)
        if pred == 1 and label == 0:
            false_positives[1] = false_positives.get(1, 0) + 1
            
    return correct_counts, total_counts, false_positives

def train_chunk_amp(model, optimizer, scaler, device, data_loader):
    """Train model on a chunk using Automatic Mixed Precision"""
    model.train()
    epoch_loss = 0
    epoch_train_f1 = 0
    
    correct_counts_sum = {}
    total_counts_sum = {}
    false_positives_sum = {}

    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        
        # Use automatic mixed precision
        with amp.autocast():
            batch_scores = model(batch_graphs, batch_x, batch_e)
            loss = model.loss(batch_scores, batch_labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.detach().item()
        
        # Get F1 score 
        from train.metrics import binary_f1_score
        epoch_train_f1 += binary_f1_score(batch_scores, batch_labels)

        # Get predictions
        preds = torch.argmax(batch_scores, dim=1)
        
        # Count edge types for this batch
        correct_counts, total_counts, false_positives = count_edge_types(preds, batch_labels)
        
        # Accumulate counts across batches
        for k, v in correct_counts.items():
            correct_counts_sum[k] = correct_counts_sum.get(k, 0) + v
        for k, v in total_counts.items():
            total_counts_sum[k] = total_counts_sum.get(k, 0) + v
        for k, v in false_positives.items():
            false_positives_sum[k] = false_positives_sum.get(k, 0) + v
            
        # Delete tensors to free GPU memory
        del batch_graphs, batch_x, batch_e, batch_labels, batch_scores, preds

    epoch_loss /= (iter + 1)
    epoch_train_f1 /= (iter + 1)
    
    return epoch_loss, epoch_train_f1, correct_counts_sum, total_counts_sum, false_positives_sum

def train_in_chunks(model, optimizer, device, dataset, params, net_params, dirs, start_epoch=0):
    """Train in small chunks to avoid memory issues."""
    t0 = time.time()
    per_epoch_time = []
    
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)
    
    # Setup parameters
    chunk_size = params.get('chunk_size', 5000)  # fallback if not in config
    epochs_per_chunk = params.get('epochs_per_chunk', 5)
    num_workers = params.get('num_workers', 4)
    pin_memory = params.get('pin_memory', True)
    use_amp = params.get('use_amp', False) and torch.cuda.is_available()
    
    # Initialize mixed precision training
    scaler = amp.GradScaler() if use_amp else None
    
    # Setup dataloaders for validation and test datasets
    trainset, valset, testset = dataset.train, dataset.val, dataset.test
    num_train_samples = len(trainset)
    num_chunks = (num_train_samples + chunk_size - 1) // chunk_size
    
    val_loader = DataLoader(valset, batch_size=params['batch_size'], 
                          shuffle=False, collate_fn=dataset.collate,
                          num_workers=num_workers, pin_memory=pin_memory)
    
    test_loader = DataLoader(testset, batch_size=1, 
                           shuffle=False, collate_fn=dataset.collate,
                           num_workers=1, pin_memory=pin_memory)
    
    # Setup learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                 factor=params['lr_reduce_factor'],
                                                 patience=params['lr_schedule_patience'],
                                                 verbose=True)
    
    # Initialize tracking variables
    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_f1s, epoch_val_f1s = [] ,[]
    total_epochs = start_epoch
    
    # Main training loop over chunks
    print(f"\nTraining on {num_chunks} chunks of size {chunk_size}")
    for chunk_idx in range(num_chunks):
        print(f"Processing chunk {chunk_idx+1}/{num_chunks}")
        
        # Create subset for this chunk
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, num_train_samples)
        train_indices = list(range(chunk_start, chunk_end))
        
        # Create dataloader for this chunk
        train_loader = DataLoader(
            Subset(trainset, train_indices),
            batch_size=params['batch_size'], 
            shuffle=True,
            collate_fn=dataset.collate,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        print(f"Chunk size: {len(train_indices)} graphs")
        
        # Train for multiple epochs on this chunk
        for epoch in range(epochs_per_chunk):
            with Timer(f"Epoch {total_epochs} (chunk {chunk_idx+1}, epoch {epoch+1})"):
                total_epochs += 1
                t_start = time.time()
                
                # Train on this chunk
                if use_amp:
                    epoch_train_loss, epoch_train_f1, train_correct_counts, train_total_counts, train_false_positives = train_chunk_amp(
                        model, optimizer, scaler, device, train_loader
                    )
                else:
                    # Use standard training function
                    from train.train_TSP_edge_classification import train_epoch_sparse as train_epoch
                    epoch_train_loss, epoch_train_f1, optimizer, train_correct_counts, train_total_counts, train_false_positives = train_epoch(
                        model, optimizer, device, train_loader, total_epochs
                    )
                
                # Evaluate on validation set
                epoch_val_loss, epoch_val_f1, val_correct_counts, val_total_counts, val_false_positives = evaluate_network(
                    model, device, val_loader, total_epochs
                )
                
                # Evaluate on test set
                epoch_test_loss, epoch_test_f1, test_correct_counts, test_total_counts, test_false_positives = evaluate_network(
                    model, device, test_loader, total_epochs
                )
                
                # Record metrics
                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_train_f1s.append(epoch_train_f1)
                epoch_val_f1s.append(epoch_val_f1)
                
                # Log to tensorboard
                writer.add_scalar('train/_loss', epoch_train_loss, total_epochs)
                writer.add_scalar('val/_loss', epoch_val_loss, total_epochs)
                writer.add_scalar('train/_f1', epoch_train_f1, total_epochs)
                writer.add_scalar('val/_f1', epoch_val_f1, total_epochs)
                writer.add_scalar('test/_f1', epoch_test_f1, total_epochs)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], total_epochs)
                
                # Calculate time
                t_end = time.time()
                per_epoch_time.append(t_end - t_start)
                
                # Print status
                print(f"Epoch {total_epochs}, Time: {t_end-t_start:.4f}s, LR: {optimizer.param_groups[0]['lr']:.6f}")
                print(f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
                print(f"Train F1: {epoch_train_f1:.4f}, Val F1: {epoch_val_f1:.4f}, Test F1: {epoch_test_f1:.4f}")
                
                # Save checkpoint
                ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                
                torch.save(model.state_dict(), f"{ckpt_dir}/epoch_{total_epochs}.pkl")
                
                # Clear older checkpoints
                files = glob.glob(f"{ckpt_dir}/*.pkl")
                for file in files:
                    epoch_nb = int(file.split('_')[-1].split('.')[0])
                    if epoch_nb < total_epochs - 1:
                        os.remove(file)
                
                # Update learning rate scheduler
                scheduler.step(epoch_val_loss)
                
                # Check for early stopping
                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET. STOPPING.")
                    break
                
                # Check for time limit
                if time.time() - t0 > params['max_time'] * 3600:
                    print(f"\nMax time ({params['max_time']} hours) elapsed. STOPPING.")
                    break
                
                # Clean up memory
                clear_memory()
        
        # Check again for time limit and min LR after each chunk
        if optimizer.param_groups[0]['lr'] < params['min_lr'] or time.time() - t0 > params['max_time'] * 3600:
            break
    
    # Final evaluation
    _, test_f1, test_correct_counts, test_total_counts, test_false_positives = evaluate_network(
        model, device, test_loader, total_epochs
    )
    
    print(f"\nFinal Test F1: {test_f1:.4f}")
    print(f"Total Epochs: {total_epochs}")
    print(f"Total Time: {(time.time()-t0)/3600:.4f} hours")
    print(f"Average Time Per Epoch: {np.mean(per_epoch_time):.4f}s")
    
    writer.close()
    
    return model

def train_val_pipeline_chunked(MODEL_NAME, dataset, params, net_params, dirs):
    """Uses train_in_chunks to avoid memory issues."""
    t0 = time.time()
    
    # Get device setup
    device = net_params['device']
    
    # Set seeds for reproducibility
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])
    
    # Initialize model and optimizer
    model = gnn_model(MODEL_NAME, net_params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    
    # Train the model
    model = train_in_chunks(model, optimizer, device, dataset, params, net_params, dirs)
    
    # Write final results and metrics
    # ...existing code...
    return model
