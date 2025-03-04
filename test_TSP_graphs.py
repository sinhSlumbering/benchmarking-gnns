import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from nets.TSP_edge_classification.load_net import gnn_model
from data.tsp_dataset import TSPDataset
import time
import argparse
import os
import pickle

from train.train_TSP_edge_classification import evaluate_network_sparse
from train.metrics import binary_f1_score

def test_graph_loading():
    """
    Test loading TSP graphs in the same format as main_TSP_edge_classification.py
    """
    # Parse arguments as done in main script
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--init_lr', type=float, default=0.001)
    parser.add_argument('--lr_reduce_factor', type=float, default=0.5)
    parser.add_argument('--lr_schedule_patience', type=int, default=5)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--print_epoch_interval', type=int, default=1)
    parser.add_argument('--max_time', type=float, default=12)
    
    # Model parameters
    parser.add_argument('--model', type=str, default="GatedGCN")
    parser.add_argument('--hidden_dim', type=int, default=70)
    parser.add_argument('--L', type=int, default=4, help="Number of layers")
    parser.add_argument('--batch_norm', type=bool, default=True)
    parser.add_argument('--residual', type=bool, default=True)
    parser.add_argument('--edge_feat', type=bool, default=True)
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--layer_type', type=str, default="edgereprfeat")
    parser.add_argument('--model_path', type=str, default="best_model.pkl", help="Path to the saved model")
    parser.add_argument('--readout', type=str, default="mean")
    parser.add_argument('--in_feat_dropout', type=float, default=0.0)
    
    # Dataset parameters
    parser.add_argument('--num_test_data', type=int, default=1000)
    parser.add_argument('--graph_size', type=int, default=100)
    
    args = parser.parse_args()
    args.device = torch.device("cuda:"+args.device) if torch.cuda.is_available() else torch.device("cpu")
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load test dataset
    print("Loading test dataset...")
    # TSPDataset loads all splits at once, so we access the test split directly
    dataset = TSPDataset("TSP")
    testset = dataset.test
    
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate)
    
    # Print some information about the loaded data
    print(f"Number of test graphs: {len(testset)}")
    sample_graph, sample_labels = testset[0]
    print(f"Sample graph - Nodes: {sample_graph.ndata['feat'].shape}, Edges: {sample_graph.edata['feat'].shape}")
    print(f"Edge labels count: {len(sample_labels)}")
    
    return test_loader, args, dataset

def load_model(args, dataset):
    """
    Load a trained model for evaluation
    """
    # Get a sample graph to determine feature dimensions
    sample_graph, _ = dataset.test[0]
    
    # Setup network parameters
    net_params = {
        'in_dim': sample_graph.ndata['feat'].shape[1],  # Node features dimension
        'in_dim_edge': sample_graph.edata['feat'].shape[1],  # Edge features dimension
        'hidden_dim': args.hidden_dim,
        'out_dim': args.hidden_dim,
        'n_classes': 2,  # Binary classification (edge is in tour or not)
        'L': args.L,
        'dropout': args.dropout,
        'batch_norm': args.batch_norm,
        'residual': args.residual,
        'edge_feat': args.edge_feat,
        'device': args.device,
        'layer_type': args.layer_type,
        'readout': args.readout,
        'in_feat_dropout': args.in_feat_dropout
    }
    
    print(f"Network parameters: {net_params}")
    
    # Initialize the model
    model = gnn_model(args.model, net_params)
    model = model.to(args.device)
    
    # Load model weights if available
    if os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    else:
        print(f"Warning: Model file {args.model_path} not found. Using untrained model.")
    
    return model

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on the test dataset
    """
    print("\nEvaluating model on test dataset...")
    model.eval()
    
    # Use the evaluate_network_sparse function from train_TSP_edge_classification.py
    test_loss, test_f1, total_predicted_as_1, total_correctly_predicted_as_1 = evaluate_network_sparse(
        model, device, test_loader, epoch=0)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Total edges predicted as type 1: {total_predicted_as_1}")
    print(f"Total edges correctly predicted as type 1: {total_correctly_predicted_as_1}")
    
    # Calculate precision and recall
    if total_predicted_as_1 > 0:
        precision = total_correctly_predicted_as_1 / total_predicted_as_1
        print(f"Precision: {precision:.4f}")
    else:
        print("Precision: N/A (no edges predicted as type 1)")
    
    # Calculate recall if possible
    total_actual_1 = 0
    for _, labels in test_loader:
        total_actual_1 += (labels == 1).sum().item()
    
    if total_actual_1 > 0:
        recall = total_correctly_predicted_as_1 / total_actual_1
        print(f"Recall: {recall:.4f}")
    else:
        print("Recall: N/A (no actual edges of type 1)")
    
    return test_loss, test_f1

if __name__ == "__main__":
    # Test graph loading
    test_loader, args, dataset = test_graph_loading()
    
    # Load model
    model = load_model(args, dataset)
    
    # Evaluate model on test dataset
    test_loss, test_f1 = evaluate_model(model, test_loader, args.device)
    
    # Example of how to iterate through the test loader
    print("\nIterating through test loader:")
    for i, (batch_graphs, batch_labels) in enumerate(test_loader):
        if i >= 2:  # Just show first 2 batches
            break
        print(f"Batch {i+1}:")
        print(f"  Batch size: {batch_graphs.batch_size}")
        print(f"  Node features: {batch_graphs.ndata['feat'].shape}")
        print(f"  Edge features: {batch_graphs.edata['feat'].shape}")
        print(f"  Edge labels: {batch_labels.shape}")
