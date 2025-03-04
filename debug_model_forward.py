import torch
import inspect
import os
import argparse
from nets.TSP_edge_classification.load_net import gnn_model
from data.tsp_dataset import TSPDataset

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Debug model forward pass')
    parser.add_argument('--model', type=str, default="GatedGCN")
    parser.add_argument('--hidden_dim', type=int, default=65)
    parser.add_argument('--L', type=int, default=4, help="Number of layers")
    parser.add_argument('--model_path', type=str, default="best_model.pkl", help="Path to the saved model")
    
    args = parser.parse_args()
    
    # Set device
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    return args

def main():
    """Debug the model's forward method"""
    args = parse_args()
    
    # Load dataset to get sample input
    print("Loading dataset...")
    dataset = TSPDataset("TSP")
    testset = dataset.test
    
    # Get a sample graph
    sample_graph, _ = testset[0]
    
    # Setup network parameters
    net_params = {
        'in_dim': sample_graph.ndata['feat'].shape[1],
        'in_dim_edge': sample_graph.edata['feat'].shape[1],
        'hidden_dim': args.hidden_dim,
        'out_dim': args.hidden_dim,
        'n_classes': 2,
        'L': args.L,
        'dropout': 0.0,
        'batch_norm': True,
        'residual': True,
        'edge_feat': True,
        'device': args.device,
        'layer_type': 'edgereprfeat',
        'readout': 'mean',
        'in_feat_dropout': 0.0
    }
    
    # Initialize model
    print("Initializing model...")
    model = gnn_model(args.model, net_params)
    model = model.to(args.device)
    
    # Print the model's architecture
    print("\nModel Architecture:")
    print(model)
    
    # Inspect the forward method
    print("\nInspecting forward method:")
    forward_sig = inspect.signature(model.forward)
    print(f"Forward method signature: {forward_sig}")
    print(f"Parameters: {list(forward_sig.parameters.keys())}")
    
    # Try to call the forward method
    print("\nTrying to call forward method...")
    sample_graph = sample_graph.to(args.device)
    h = sample_graph.ndata['feat'].to(args.device)
    e = sample_graph.edata['feat'].to(args.device)
    
    with torch.no_grad():
        try:
            # Try with just graph
            print("Attempt 1: model(graph)")
            try:
                output = model(sample_graph)
                print("Success!")
            except Exception as exc:
                print(f"Failed: {exc}")
            
            # Try with graph, h, e
            print("\nAttempt 2: model(graph, h, e)")
            try:
                output = model(sample_graph, h, e)
                print("Success!")
                print(f"Output shape: {output.shape}")
                
                # Print prediction example
                _, predicted = torch.max(output, 1)
                print(f"Example predictions (first 10): {predicted[:10]}")
            except Exception as exc:
                print(f"Failed: {exc}")
        except Exception as e:
            print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
