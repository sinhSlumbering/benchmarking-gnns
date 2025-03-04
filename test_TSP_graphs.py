import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from nets.TSP_edge_classification.load_net import gnn_model
from data.TSP import TSPDataset
import time
import argparse

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
    parser.add_argument('--num_node_features', type=int, default=1)
    parser.add_argument('--num_edge_features', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=70)
    parser.add_argument('--num_layers', type=int, default=10)
    parser.add_argument('--batch_norm', type=bool, default=True)
    parser.add_argument('--residual', type=bool, default=True)
    parser.add_argument('--edge_feat', type=bool, default=True)
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--in_dim_edge', type=int, default=1)
    parser.add_argument('--out_dim_edge', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    
    # Dataset parameters
    parser.add_argument('--num_train_data', type=int, default=10000)
    parser.add_argument('--num_valid_data', type=int, default=1000)
    parser.add_argument('--num_test_data', type=int, default=1000)
    
    args = parser.parse_args()
    args.device = torch.device("cuda:"+args.device) if torch.cuda.is_available() else torch.device("cpu")
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load test dataset
    print("Loading test dataset...")
    testset = TSPDataset(args.num_test_data, data_dir='data/TSP', split='test')
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    
    # Print some information about the loaded data
    print(f"Number of test graphs: {len(testset)}")
    sample = testset[0]
    print(f"Sample graph - Nodes: {sample.x.shape}, Edges: {sample.edge_index.shape}")
    print(f"Edge features: {sample.edge_attr.shape}")
    print(f"Edge labels: {sample.edge_label.shape}")
    
    return test_loader

if __name__ == "__main__":
    # Test graph loading
    test_loader = test_graph_loading()
    
    # Example of how to iterate through the test loader
    print("\nIterating through test loader:")
    for i, data in enumerate(test_loader):
        if i >= 2:  # Just show first 2 batches
            break
        print(f"Batch {i+1}:")
        print(f"  Batch size: {data.num_graphs}")
        print(f"  Node features: {data.x.shape}")
        print(f"  Edge index: {data.edge_index.shape}")
        print(f"  Edge features: {data.edge_attr.shape}")
        print(f"  Edge labels: {data.edge_label.shape}")
