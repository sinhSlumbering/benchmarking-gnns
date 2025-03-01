"""
    IMPORTING LIBS
"""
import dgl

import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm

import torch.cuda.amp as amp 
from optimize_training import clear_memory, print_memory_stats, Timer

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self
        
"""
    IMPORTING CUSTOM MODULES/METHODS
"""
from nets.TSP_edge_classification.load_net import gnn_model # import all GNNS
from data.load_data import LoadData # import dataset
from data.stream_data import StreamingTSPDataset # import streaming dataset
from train_val_pipeline_chunked import train_val_pipeline_chunked


"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


"""
    VIEWING MODEL CONFIG AND PARAMS
"""
def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


"""
    Helper functions for metrics calculation
"""
def count_edge_types(preds, labels):
    """
    Count correct predictions, total counts, and false positives by edge type
    """
    correct_counts = {}
    total_counts = {}
    false_positives = {}
    
    for i in range(labels.shape[0]):
        label = labels[i].item()
        pred = preds[i].item()
        
        # Count total for each class
        if label not in total_counts:
            total_counts[label] = 0
        total_counts[label] += 1
        
        # Count correct predictions
        if pred == label:
            if label not in correct_counts:
                correct_counts[label] = 0
            correct_counts[label] += 1
        else:
            # Count false positives for predicted class
            if pred not in false_positives:
                false_positives[pred] = 0
            false_positives[pred] += 1
    
    return correct_counts, total_counts, false_positives

def binary_f1_score(scores, labels):
    """
    Calculate F1 score for binary classification
    """
    preds = torch.argmax(scores, dim=1)
    
    # For binary classification, we're interested in the positive class (usually 1)
    true_positives = ((preds == 1) & (labels == 1)).sum().item()
    false_positives = ((preds == 1) & (labels == 0)).sum().item()
    false_negatives = ((preds == 0) & (labels == 1)).sum().item()
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1


"""
    TRAINING CODE
"""

def create_optimized_dataloaders(dataset, batch_size, collate_fn, num_workers=4, pin_memory=True):
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        dataset.train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        dataset.val,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    
    test_loader = DataLoader(
        dataset.test,
        batch_size=1,  # Keep batch_size=1 for testing
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=1,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader

def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs):
    torch.cuda.empty_cache()
    t0 = time.time()
    per_epoch_time = []
    
    DATASET_NAME = dataset.name
    is_streaming = isinstance(dataset, StreamingTSPDataset)
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']
    
    # Print initial memory usage
    print("Initial memory usage:")
    print_memory_stats()
    
    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""
                .format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))
        
    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)

    # Setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])
    
    # Create model
    model = gnn_model(MODEL_NAME, net_params)
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                    factor=params['lr_reduce_factor'],
                                                    patience=params['lr_schedule_patience'],
                                                    verbose=True)
    
    # Setup AMP for mixed precision training if enabled
    use_amp = params.get('use_amp', False) and torch.cuda.is_available()
    scaler = amp.GradScaler() if use_amp else None
    
    # Create optimized dataloaders
    num_workers = params.get('num_workers', 0)
    pin_memory = params.get('pin_memory', False)
    
    if MODEL_NAME in ['RingGNN', '3WLGNN']:
        # Import train functions specific for WL-GNNs
        from train.train_TSP_edge_classification import train_epoch_dense as train_epoch, evaluate_network_dense as evaluate_network
        from functools import partial # util function to pass edge_feat to collate function
        
        train_loader = DataLoader(dataset.train, shuffle=True, collate_fn=partial(dataset.collate_dense_gnn, edge_feat=net_params['edge_feat']))
        val_loader = DataLoader(dataset.val, shuffle=False, collate_fn=partial(dataset.collate_dense_gnn, edge_feat=net_params['edge_feat']))
        test_loader = DataLoader(dataset.test, shuffle=False, collate_fn=partial(dataset.collate_dense_gnn, edge_feat=net_params['edge_feat']))
    else:
        # Import train functions for all other GCNs
        from train.train_TSP_edge_classification import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network
        train_loader, val_loader, test_loader = create_optimized_dataloaders(
            dataset, params['batch_size'], dataset.collate, num_workers, pin_memory
        )
    
    # Track losses and F1 scores
    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_f1s, epoch_val_f1s = [], []
    
    # Define the starting epoch
    start_epoch = 0
    
    # Main training loop
    try:
        with tqdm(range(start_epoch, params['epochs'])) as t:
            for epoch in t:
                t.set_description(f'Epoch {epoch}')
                start = time.time()
                
                # Training with AMP support
                with Timer("Training"):
                    if use_amp and MODEL_NAME not in ['RingGNN', '3WLGNN']:
                        # AMP training loop
                        model.train()
                        epoch_loss = 0
                        epoch_train_f1 = 0
                        correct_counts_sum = {}
                        total_counts_sum = {}
                        false_positives_sum = {}
                        
                        for iter, (batch_graphs, batch_labels) in enumerate(train_loader):
                            batch_graphs = batch_graphs.to(device)
                            batch_x = batch_graphs.ndata['feat'].to(device)
                            batch_e = batch_graphs.edata['feat'].to(device)
                            batch_labels = batch_labels.to(device)
                            
                            optimizer.zero_grad()
                            with amp.autocast():
                                batch_scores = model(batch_graphs, batch_x, batch_e)
                                loss = model.loss(batch_scores, batch_labels)
                            
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                            
                            # Calculate metrics
                            preds = torch.argmax(batch_scores, dim=1)
                            correct_counts, total_counts, false_positives = count_edge_types(preds, batch_labels)
                            
                            # Update summary counts
                            for key, val in correct_counts.items():
                                if key not in correct_counts_sum:
                                    correct_counts_sum[key] = 0
                                correct_counts_sum[key] += val
                                
                            for key, val in total_counts.items():
                                if key not in total_counts_sum:
                                    total_counts_sum[key] = 0
                                total_counts_sum[key] += val
                                
                            for key, val in false_positives.items():
                                if key not in false_positives_sum:
                                    false_positives_sum[key] = 0
                                false_positives_sum[key] += val
                            
                            # Accumulate results
                            epoch_loss += loss.detach().item()
                            epoch_train_f1 += binary_f1_score(batch_scores, batch_labels)
                            
                            # Free memory
                            del batch_graphs, batch_x, batch_e, batch_scores, batch_labels, preds
                            
                        # Normalize results
                        epoch_loss /= (iter + 1)
                        epoch_train_f1 /= (iter + 1)
                        
                        # Clear memory after training
                        clear_memory()
                        
                        epoch_train_loss = epoch_loss
                        train_correct_counts = correct_counts_sum
                        train_total_counts = total_counts_sum
                        train_false_positives = false_positives_sum
                    else:
                        # Standard training based on model type
                        if MODEL_NAME in ['RingGNN', '3WLGNN']:
                            epoch_train_loss, epoch_train_f1, optimizer = train_epoch(
                                model, optimizer, device, train_loader, epoch, params['batch_size'])
                        else:
                            epoch_train_loss, epoch_train_f1, optimizer = train_epoch(
                                model, optimizer, device, train_loader, epoch)
                        
                        # Get detailed train metrics
                        train_metrics = evaluate_network(model, device, train_loader, epoch)
                        epoch_train_loss = train_metrics['loss']
                        epoch_train_f1 = train_metrics['f1']
                        train_correct_counts = train_metrics.get('correct_counts', {})
                        train_total_counts = train_metrics.get('total_counts', {})
                        train_false_positives = train_metrics.get('false_positives', {})
                
                # Validation and testing
                with Timer("Validation"):
                    val_metrics = evaluate_network(model, device, val_loader, epoch)
                    epoch_val_loss = val_metrics['loss']
                    epoch_val_f1 = val_metrics['f1']
                    val_correct_counts = val_metrics.get('correct_counts', {})
                    val_total_counts = val_metrics.get('total_counts', {})
                    val_false_positives = val_metrics.get('false_positives', {})
                
                with Timer("Testing"):
                    test_metrics = evaluate_network(model, device, test_loader, epoch)
                    epoch_test_loss = test_metrics['loss']
                    epoch_test_f1 = test_metrics['f1']
                    test_correct_counts = test_metrics.get('correct_counts', {})
                    test_total_counts = test_metrics.get('total_counts', {})
                    test_false_positives = test_metrics.get('false_positives', {})
                    test_correct_optimal = test_metrics.get('correct_optimal_edges', 0)
                    test_total_optimal = test_metrics.get('total_optimal_edges', 0)
                    test_total_predicted = test_metrics.get('total_predicted_edges', 0)
                
                # Record metrics
                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_train_f1s.append(epoch_train_f1)
                epoch_val_f1s.append(epoch_val_f1)
                
                # Log metrics to TensorBoard
                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                writer.add_scalar('train/_f1', epoch_train_f1, epoch)
                writer.add_scalar('val/_f1', epoch_val_f1, epoch)
                writer.add_scalar('test/_f1', epoch_test_f1, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
                
                # Print memory usage occasionally
                if epoch % 10 == 0:
                    print(f"\nMemory usage after epoch {epoch}:")
                    print_memory_stats()
                
                # Update progress bar
                t.set_postfix(
                    time=time.time()-start,
                    lr=optimizer.param_groups[0]['lr'],
                    train_loss=epoch_train_loss,
                    val_loss=epoch_val_loss,
                    train_f1=epoch_train_f1,
                    val_f1=epoch_val_f1,
                    test_f1=epoch_test_f1
                )
                
                # Free up memory
                clear_memory()
                
                per_epoch_time.append(time.time()-start)
                
                # Saving checkpoint
                ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))
                
                # Clean up older checkpoints
                files = glob.glob(ckpt_dir + '/*.pkl')
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch-1:
                        os.remove(file)
                
                # Update learning rate
                scheduler.step(epoch_val_loss)
                
                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break
                
                # Stop training after params['max_time'] hours
                if time.time()-t0 > params['max_time']*3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break
                
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    
    # Final evaluation on test set
    final_test_metrics = evaluate_network(model, device, test_loader, epoch)
    final_test_f1 = final_test_metrics['f1']
    final_correct_optimal = final_test_metrics.get('correct_optimal_edges', 0)
    final_total_optimal = final_test_metrics.get('total_optimal_edges', 0)  
    final_total_predicted = final_test_metrics.get('total_predicted_edges', 0)
    
    # Final evaluation on train set
    final_train_metrics = evaluate_network(model, device, train_loader, epoch)
    train_f1 = final_train_metrics['f1']
    
    print("Test F1: {:.4f}".format(final_test_f1))
    print("Train F1: {:.4f}".format(train_f1))
    print("Correctly Identified Optimal Edges: {}".format(final_correct_optimal))
    print("Total Optimal Edges: {}".format(final_total_optimal))
    print("Total Predicted Optimal Edges: {}".format(final_total_predicted))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))
    
    # Close TensorBoard writer
    writer.close()
    
    # Write the results to the output file
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS
    TEST F1: {:.4f}
    TRAIN F1: {:.4f}

    Convergence Time (Epochs): {:.4f}
    Total Time Taken: {:.4f}hrs
    Average Time Per Epoch: {:.4f}s

    Additional Metrics:
    Correctly Identified Optimal Edges: {}
    Total Optimal Edges: {}
    Total Predicted Optimal Edges: {}

    """.format(
            DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
            final_test_f1, train_f1,
            epoch, (time.time()-t0)/3600, np.mean(per_epoch_time),
            final_correct_optimal, final_total_optimal, final_total_predicted
        ))

def train_val_pipeline_chunked(MODEL_NAME, dataset, params, net_params, dirs):
    torch.cuda.empty_cache()
    t0 = time.time()
    per_epoch_time = []
        
    DATASET_NAME = dataset.name
    trainset, valset, testset = dataset.train, dataset.val, dataset.test
        
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']
    
    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\nTotal Parameters: {}\n\n"""                .format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))
        
    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)

    # Setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])
    
    print("Total Training Graphs: ", len(trainset))
    print("Validation Graphs: ", len(valset))
    print("Test Graphs: ", len(testset))
    print("Number of Classes: ", net_params['n_classes'])

    model = gnn_model(MODEL_NAME, net_params)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)
    
    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_f1s, epoch_val_f1s = [], [] 
    
    # Import train functions for GNNs
    from train.train_TSP_edge_classification import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network

    val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    
    # Now, split trainset into chunks of size 10000
    num_train_samples = len(trainset)
    chunk_size = 10000
    num_chunks = (num_train_samples + chunk_size - 1) // chunk_size
    epochs_per_chunk = 10  # Number of epochs to train per chunk

    total_epochs = 0

    for chunk_idx in range(num_chunks):
        print("Processing chunk {}/{}".format(chunk_idx+1, num_chunks))

        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, num_train_samples)
        train_indices = list(range(chunk_start, chunk_end))
        from torch.utils.data import Subset
        trainset_chunk = Subset(trainset, train_indices)

        train_loader = DataLoader(trainset_chunk, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.collate)

        print("Number of graphs in chunk:", len(trainset_chunk))

        # Train for epochs_per_chunk epochs or until convergence
        for epoch in range(epochs_per_chunk):

            t_start = time.time()

            total_epochs += 1

            # Training - adjust to match the actual return values
            epoch_train_loss, epoch_train_f1, optimizer, train_metrics = train_epoch(
                model, optimizer, device, train_loader, epoch)
            
            train_correct_counts = train_metrics.get('correct_counts', {})
            train_total_counts = train_metrics.get('total_counts', {})
            train_false_positives = train_metrics.get('false_positives', {})
            
            # Validation
            epoch_val_loss, epoch_val_f1, val_metrics = evaluate_network(
                model, device, val_loader, epoch)
                
            val_correct_counts = val_metrics.get('correct_counts', {})
            val_total_counts = val_metrics.get('total_counts', {})
            val_false_positives = val_metrics.get('false_positives', {})
            
            # Testing
            epoch_test_loss, epoch_test_f1, test_metrics = evaluate_network(
                model, device, test_loader, epoch)
                
            test_correct_counts = test_metrics.get('correct_counts', {})
            test_total_counts = test_metrics.get('total_counts', {})
            test_false_positives = test_metrics.get('false_positives', {})
            
            epoch_train_losses.append(epoch_train_loss)
            epoch_val_losses.append(epoch_val_loss)
            epoch_train_f1s.append(epoch_train_f1)
            epoch_val_f1s.append(epoch_val_f1)

            writer.add_scalar('train/_loss', epoch_train_loss, total_epochs)
            writer.add_scalar('val/_loss', epoch_val_loss, total_epochs)
            writer.add_scalar('train/_f1', epoch_train_f1, total_epochs)
            writer.add_scalar('val/_f1', epoch_val_f1, total_epochs)
            writer.add_scalar('test/_f1', epoch_test_f1, total_epochs)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], total_epochs)

            t_end = time.time()

            per_epoch_time.append(t_end - t_start)

            print("Epoch {}, Time: {:.4f}, LR: {:.6f}, Train Loss: {:.4f}, Val Loss: {:.4f}, Train F1: {:.4f}, Val F1: {:.4f}, Test F1: {:.4f}".format(
                total_epochs, t_end - t_start, optimizer.param_groups[0]['lr'], epoch_train_loss, epoch_val_loss, epoch_train_f1, epoch_val_f1, epoch_test_f1))

            # Saving checkpoint
            ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(total_epochs)))

            files = glob.glob(ckpt_dir + '/*.pkl')
            for file in files:
                epoch_nb = file.split('_')[-1]
                epoch_nb = int(epoch_nb.split('.')[0])
                if epoch_nb < total_epochs - 1:
                    os.remove(file)

            scheduler.step(epoch_val_loss)

            if optimizer.param_groups[0]['lr'] < params['min_lr']:
                print("\n!! LR EQUAL TO MIN LR SET.")
                break

            # Stop training after params['max_time'] hours
            if time.time() - t0 > params['max_time'] * 3600:
                print('-' * 89)
                print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                break

    # Final evaluation on test set
    _, test_f1, test_metrics = evaluate_network(model, device, test_loader, epoch)
    test_correct_counts = test_metrics.get('correct_counts', {})
    test_total_counts = test_metrics.get('total_counts', {}) 
    test_false_positives = test_metrics.get('false_positives', {})
    
    _, train_f1, train_metrics = evaluate_network(model, device, train_loader, epoch)
    train_correct_counts = train_metrics.get('correct_counts', {})
    train_total_counts = train_metrics.get('total_counts', {})
    train_false_positives = train_metrics.get('false_positives', {})
    
    print("Test F1: {:.4f}".format(test_f1))
    print("Train F1: {:.4f}".format(train_f1))
    print("Convergence Time (Epochs): {:.4f}".format(total_epochs))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    writer.close()

    # Write the results
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST F1: {:.4f}\nTRAIN F1: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f}hrs\nAverage Time Per Epoch: {:.4f}s\n\n\n"""\
                .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                        test_f1, train_f1, total_epochs, (time.time() - t0) / 3600, np.mean(per_epoch_time)))


def main():    
    """
        USER CONTROLS
    """
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")    
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--kernel', help="Please give a value for kernel")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--gated', help="Please give a value for gated")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--layer_norm', help="Please give a value for layer_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--sage_aggregator', help="Please give a value for sage_aggregator")
    parser.add_argument('--data_mode', help="Please give a value for data_mode")
    parser.add_argument('--num_pool', help="Please give a value for num_pool")
    parser.add_argument('--gnn_per_block', help="Please give a value for gnn_per_block")
    parser.add_argument('--embedding_dim', help="Please give a value for embedding_dim")
    parser.add_argument('--pool_ratio', help="Please give a value for pool_ratio")
    parser.add_argument('--linkpred', help="Please give a value for linkpred")
    parser.add_argument('--cat', help="Please give a value for cat")
    parser.add_argument('--self_loop', help="Please give a value for self_loop")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    parser.add_argument('--layer_type', help="Please give a value for layer_type (for GAT and GatedGCN only)")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
        
    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    dataset = LoadData(DATASET_NAME)
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)
    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)   
    if args.residual is not None:
        net_params['residual'] = True if args.residual=='True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat=='True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.kernel is not None:
        net_params['kernel'] = int(args.kernel)
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.gated is not None:
        net_params['gated'] = True if args.gated=='True' else False
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.layer_norm is not None:
        net_params['layer_norm'] = True if args.layer_norm=='True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm=='True' else False
    if args.sage_aggregator is not None:
        net_params['sage_aggregator'] = args.sage_aggregator
    if args.data_mode is not None:
        net_params['data_mode'] = args.data_mode
    if args.num_pool is not None:
        net_params['num_pool'] = int(args.num_pool)
    if args.gnn_per_block is not None:
        net_params['gnn_per_block'] = int(args.gnn_per_block)
    if args.embedding_dim is not None:
        net_params['embedding_dim'] = int(args.embedding_dim)
    if args.pool_ratio is not None:
        net_params['pool_ratio'] = float(args.pool_ratio)
    if args.linkpred is not None:
        net_params['linkpred'] = True if args.linkpred=='True' else False
    if args.cat is not None:
        net_params['cat'] = True if args.cat=='True' else False
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop=='True' else False
    if args.layer_type is not None:
        net_params['layer_type'] = args.layer_type
 

      
    
    # TSP
    net_params['in_dim'] = dataset.train[0][0].ndata['feat'][0].shape[0]
    net_params['in_dim_edge'] = dataset.train[0][0].edata['feat'][0].size(0)
    num_classes = len(np.unique(np.concatenate(dataset.train[:][1])))
    net_params['n_classes'] = num_classes
    
    if MODEL_NAME == 'RingGNN':
        num_nodes = [dataset.train[i][0].number_of_nodes() for i in range(len(dataset.train))]
        net_params['avg_node_num'] = int(np.ceil(np.mean(num_nodes)))
    
    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')
        
    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
    
    # Toggle between standard and chunked training
    if params.get('use_chunked_training', False):
        train_val_pipeline_chunked(MODEL_NAME, dataset, params, net_params, dirs)
    else:
        train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs)

main()
