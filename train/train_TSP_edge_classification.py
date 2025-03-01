"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import dgl

from train.metrics import binary_f1_score

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

"""
    For GCNs
"""
def train_epoch_sparse(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_f1 = 0
    nb_data = 0
    gpu_mem = 0
    
    correct_counts_sum = {}
    total_counts_sum = {}
    false_positives_sum = {}

    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # Node features
        batch_e = batch_graphs.edata['feat'].to(device)  # Edge features
        batch_labels = batch_labels.to(device)  # Edge labels
        optimizer.zero_grad()
        
        batch_scores = model.forward(batch_graphs, batch_x, batch_e)
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
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

    epoch_loss /= (iter + 1)
    epoch_train_f1 /= (iter + 1)
    
    return epoch_loss, epoch_train_f1, optimizer, correct_counts_sum, total_counts_sum, false_positives_sum


def evaluate_network_sparse(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_f1 = 0
    nb_data = 0
    
    correct_counts_sum = {}
    total_counts_sum = {}
    false_positives_sum = {}

    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_labels = batch_labels.to(device)

            batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            loss = model.loss(batch_scores, batch_labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_f1 += binary_f1_score(batch_scores, batch_labels)

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
                
    epoch_test_loss /= (iter + 1)
    epoch_test_f1 /= (iter + 1)
    
    return epoch_test_loss, epoch_test_f1, correct_counts_sum, total_counts_sum, false_positives_sum


"""
    For WL-GNNs
"""
def train_epoch_dense(model, optimizer, device, data_loader, epoch, batch_size):

    model.train()
    epoch_loss = 0
    epoch_train_f1 = 0
    nb_data = 0
    gpu_mem = 0
    optimizer.zero_grad()
    for iter, (x_no_edge_feat, x_with_edge_feat, labels, edge_list) in enumerate(data_loader):
        if x_no_edge_feat is not None:
            x_no_edge_feat = x_no_edge_feat.to(device)
        if x_with_edge_feat is not None:
            x_with_edge_feat = x_with_edge_feat.to(device)
        labels = labels.to(device)
        edge_list = edge_list[0].to(device), edge_list[1].to(device)
        
        scores = model.forward(x_no_edge_feat, x_with_edge_feat, edge_list)
        loss = model.loss(scores, labels)
        loss.backward()
        
        if not (iter%batch_size):
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_loss += loss.detach().item()
        epoch_train_f1 += binary_f1_score(scores, labels)
    epoch_loss /= (iter + 1)
    epoch_train_f1 /= (iter + 1)
    
    return epoch_loss, epoch_train_f1, optimizer


def evaluate_network_dense(model, device, data_loader, epoch):
    
    model.eval()
    epoch_test_loss = 0
    epoch_test_f1 = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (x_no_edge_feat, x_with_edge_feat, labels, edge_list) in enumerate(data_loader):
            if x_no_edge_feat is not None:
                x_no_edge_feat = x_no_edge_feat.to(device)
            if x_with_edge_feat is not None:
                x_with_edge_feat = x_with_edge_feat.to(device)
            labels = labels.to(device)
            edge_list = edge_list[0].to(device), edge_list[1].to(device)

            scores = model.forward(x_no_edge_feat, x_with_edge_feat, edge_list)
            loss = model.loss(scores, labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_f1 += binary_f1_score(scores, labels)
        epoch_test_loss /= (iter + 1)
        epoch_test_f1 /= (iter + 1)
        
    return epoch_test_loss, epoch_test_f1
