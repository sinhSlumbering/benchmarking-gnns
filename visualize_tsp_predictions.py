import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import random
import networkx as nx
from pathlib import Path

from nets.TSP_edge_classification.load_net import gnn_model
from data.tsp_dataset import TSPDataset
from torch.utils.data import DataLoader

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize TSP Predictions')
    
    # Model parameters
    parser.add_argument('--model', type=str, default="GatedGCN")
    parser.add_argument('--hidden_dim', type=int, default=65)  # Match the saved model
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
    
    # Visualization parameters
    parser.add_argument('--graph_index', type=int, default=None, help="Index of the graph to visualize. Random if None.")
    parser.add_argument('--output_dir', type=str, default="./tsp_visualizations", help="Directory to save visualizations")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set device
    if torch.cuda.is_available() and args.device != "cpu":
        args.device = torch.device(f"cuda:{args.device}")
    else:
        args.device = torch.device("cpu")
    
    return args

def load_model_and_data(args):
    """Load the trained model and test dataset."""
    print("Loading dataset...")
    dataset = TSPDataset("TSP")
    testset = dataset.test
    
    print(f"Number of test graphs: {len(testset)}")
    
    # Get a sample graph to determine dimensions
    sample_graph, _ = testset[0]
    
    # Setup network parameters
    net_params = {
        'in_dim': sample_graph.ndata['feat'].shape[1],
        'in_dim_edge': sample_graph.edata['feat'].shape[1],
        'hidden_dim': args.hidden_dim,
        'out_dim': args.hidden_dim,
        'n_classes': 2,
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
    
    # Initialize and load the model
    print(f"Loading model from {args.model_path}...")
    model = gnn_model(args.model, net_params)
    model = model.to(args.device)
    
    if os.path.exists(args.model_path):
        try:
            checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=True)
            if 'criterion.weight' in checkpoint:
                del checkpoint['criterion.weight']
            model.load_state_dict(checkpoint, strict=False)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None, None
    else:
        print(f"Model file {args.model_path} not found")
        return None, None
    
    return model, testset

def select_graph(testset, args):
    """Select a graph to visualize."""
    if args.graph_index is not None and 0 <= args.graph_index < len(testset):
        index = args.graph_index
    else:
        # Set random seed for reproducibility
        random.seed(args.seed)
        index = random.randint(0, len(testset) - 1)
    
    print(f"Selected graph index: {index}")
    return testset[index], index

def make_predictions(model, graph, args):
    """Make predictions on the selected graph."""
    model.eval()
    with torch.no_grad():
        # Prepare graph
        graph = graph.to(args.device)
        
        # Forward pass
        pred = model(graph)
        
        # Get edge predictions (binary classification)
        _, predicted = torch.max(pred, 1)
        
        return predicted

def visualize_tsp_solution(graph, edge_labels, predictions, graph_index, args):
    """Visualize the TSP solution with color-coded edges."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert DGL graph to NetworkX for visualization
    g_nx = graph.to_networkx(node_attrs=['feat'], edge_attrs=['feat'])
    
    # Get node positions from node features (x,y coordinates)
    pos = {}
    for node_id, node_data in g_nx.nodes(data=True):
        # Node features contain the x,y coordinates
        pos[node_id] = node_data['feat'].cpu().numpy()
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(g_nx, pos, node_size=300, node_color='lightblue', alpha=0.8)
    
    # Draw edges with color coding:
    # - Green: Predicted correctly
    # - Red: Predicted incorrectly
    
    # Create lists for edges based on prediction results
    edges_correct = []
    edges_incorrect = []
    
    # Get original edge order to match predictions with edges
    edge_list = list(g_nx.edges())
    
    for i, (u, v) in enumerate(edge_list):
        # Convert to CPU tensors
        pred = predictions[i].cpu().item()
        label = edge_labels[i].cpu().item()
        
        if pred == label:
            edges_correct.append((u, v))
        else:
            edges_incorrect.append((u, v))
    
    # Draw correct edges in green
    nx.draw_networkx_edges(g_nx, pos, edgelist=edges_correct, width=1.5, alpha=0.7, edge_color='green')
    
    # Draw incorrect edges in red
    nx.draw_networkx_edges(g_nx, pos, edgelist=edges_incorrect, width=1.5, alpha=0.7, edge_color='red')
    
    # Calculate metrics for this graph
    total_edges = len(edge_list)
    correct_edges = len(edges_correct)
    incorrect_edges = len(edges_incorrect)
    accuracy = correct_edges / total_edges if total_edges > 0 else 0
    
    # Add a title with metrics
    plt.title(f"TSP Graph {graph_index} - Prediction Results\n"
              f"Total Edges: {total_edges}, Correct: {correct_edges}, "
              f"Incorrect: {incorrect_edges}, Accuracy: {accuracy:.4f}", 
              fontsize=12)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Correct Prediction'),
        Line2D([0], [0], color='red', lw=2, label='Incorrect Prediction')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Remove axis
    plt.axis('off')
    
    # Save visualization
    file_path = os.path.join(args.output_dir, f"tsp_graph_{graph_index}_visualization.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {file_path}")
    
    # Show the plot
    plt.show()
    
    return accuracy, correct_edges, incorrect_edges

def create_actual_tour_visualization(graph, edge_labels, graph_index, args):
    """Create visualization of the actual TSP tour."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert DGL graph to NetworkX for visualization
    g_nx = graph.to_networkx(node_attrs=['feat'], edge_attrs=['feat'])
    
    # Get node positions from node features (x,y coordinates)
    pos = {}
    for node_id, node_data in g_nx.nodes(data=True):
        pos[node_id] = node_data['feat'].cpu().numpy()
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Draw all nodes
    nx.draw_networkx_nodes(g_nx, pos, node_size=300, node_color='lightblue', alpha=0.8)
    
    # Get edges that are part of the tour (label = 1)
    edge_list = list(g_nx.edges())
    tour_edges = []
    for i, (u, v) in enumerate(edge_list):
        if edge_labels[i].cpu().item() == 1:
            tour_edges.append((u, v))
    
    # Draw tour edges in blue
    nx.draw_networkx_edges(g_nx, pos, edgelist=tour_edges, width=2.0, alpha=0.9, edge_color='blue')
    
    # Add title
    plt.title(f"TSP Graph {graph_index} - Actual Tour", fontsize=12)
    
    # Remove axis
    plt.axis('off')
    
    # Save visualization
    file_path = os.path.join(args.output_dir, f"tsp_graph_{graph_index}_actual_tour.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Actual tour visualization saved to {file_path}")
    
    # Show the plot
    plt.show()

def main():
    """Main function."""
    args = parse_args()
    
    # Load model and data
    model, testset = load_model_and_data(args)
    if model is None or testset is None:
        print("Failed to load model or dataset. Exiting.")
        return
    
    # Select a graph to visualize
    (graph, edge_labels), graph_index = select_graph(testset, args)
    
    # Make predictions
    predictions = make_predictions(model, graph, args)
    
    # First create visualization of actual tour
    create_actual_tour_visualization(graph, edge_labels, graph_index, args)
    
    # Visualize predictions with color-coded edges
    accuracy, correct_edges, incorrect_edges = visualize_tsp_solution(
        graph, edge_labels, predictions, graph_index, args)
    
    print(f"Graph {graph_index} prediction accuracy: {accuracy:.4f}")
    print(f"Correct edges: {correct_edges}, Incorrect edges: {incorrect_edges}")

if __name__ == "__main__":
    main()
