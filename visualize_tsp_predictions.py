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

def process_undirected_predictions(graph, predictions):
    """Process predictions to ensure consistent undirected graph representation.
    For conflicting u->v and v->u predictions, choose the one that will increase
    the number of predicted 1 edges."""
    # Get the graph's edges and their reverse mappings
    g_nx = graph.to_networkx(node_attrs=['feat'], edge_attrs=['feat'])
    edge_list = list(g_nx.edges())
    
    # Create a dictionary to store edge predictions
    edge_pred_dict = {}
    
    # First pass: collect all predictions
    for i, (u, v) in enumerate(edge_list):
        pred = predictions[i].cpu().item()
        # Store both directions
        edge_pred_dict[(u, v)] = pred
        
    # Second pass: resolve conflicts for undirected representation
    resolved_predictions = torch.clone(predictions)
    
    for i, (u, v) in enumerate(edge_list):
        # Check if reverse edge exists in our dictionary
        if (v, u) in edge_pred_dict and (u, v) in edge_pred_dict:
            # If predictions differ, prefer the one with value 1
            if edge_pred_dict[(u, v)] != edge_pred_dict[(v, u)]:
                # Choose prediction 1 over 0
                resolved_predictions[i] = 1
    
    return resolved_predictions

def make_predictions(model, graph, args):
    """Make predictions on the selected graph."""
    model.eval()
    with torch.no_grad():
        # Prepare graph
        graph = graph.to(args.device)
        
        # Extract node and edge features
        h = graph.ndata['feat']
        e = graph.edata['feat']
        
        # Forward pass with node and edge features
        pred = model(graph, h, e)
        
        # Get edge predictions (binary classification)
        _, predicted = torch.max(pred, 1)
        
        # Process predictions to ensure undirected graph consistency
        predicted = process_undirected_predictions(graph, predicted)
        
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
    # - Light Grey: Edges predicted as 1 (part of sparsified graph)
    # - Green: Correctly predicted edges that are part of the actual TSP solution (label=1, pred=1)
    # - Red: Incorrectly predicted edges that should have been part of the solution (label=1, pred=0)
    
    # Create lists for different edge categories
    edges_predicted_as_1 = []  # All edges predicted as 1 (sparsified graph)
    edges_correct_solution = []  # Correctly predicted edges that are part of the solution (label=1, pred=1)
    edges_missed_solution = []  # Edges that should be in solution but were missed (label=1, pred=0)
    
    # Get original edge order to match predictions with edges
    edge_list = list(g_nx.edges())
    
    for i, (u, v) in enumerate(edge_list):
        # Convert to CPU tensors
        pred = predictions[i].cpu().item()
        
        # Check if edge_labels is a tensor or numpy array and handle accordingly
        if isinstance(edge_labels[i], torch.Tensor):
            label = edge_labels[i].cpu().item()
        else:
            # Handle numpy or other types directly
            label = edge_labels[i]
        
        # Edges predicted as 1 (part of sparsified graph)
        if pred == 1:
            edges_predicted_as_1.append((u, v))
            
            # If it's also part of the actual solution (label=1)
            if label == 1:
                edges_correct_solution.append((u, v))
        
        # Edges that should be in solution but were missed (label=1, pred=0)
        elif label == 1 and pred == 0:
            edges_missed_solution.append((u, v))
    
    # Draw edges predicted as 1 in light grey (sparsified graph)
    nx.draw_networkx_edges(g_nx, pos, edgelist=edges_predicted_as_1, width=1.5, alpha=0.5, edge_color='lightgrey')
    
    # Draw correctly predicted solution edges in green (overwrite the grey ones)
    nx.draw_networkx_edges(g_nx, pos, edgelist=edges_correct_solution, width=2.0, alpha=0.9, edge_color='green')
    
    # Draw missed solution edges in red
    nx.draw_networkx_edges(g_nx, pos, edgelist=edges_missed_solution, width=2.0, alpha=0.9, edge_color='red', style='dashed')
    
    # Calculate metrics for this graph
    total_predicted_as_1 = len(edges_predicted_as_1)
    total_actual_solution = sum(1 for i, _ in enumerate(edge_list) if 
                              (isinstance(edge_labels[i], torch.Tensor) and edge_labels[i].cpu().item() == 1) or 
                              (not isinstance(edge_labels[i], torch.Tensor) and edge_labels[i] == 1))
    correct_solution_edges = len(edges_correct_solution)
    missed_solution_edges = len(edges_missed_solution)
    
    solution_accuracy = correct_solution_edges / total_actual_solution if total_actual_solution > 0 else 0
    
    # Add a title with metrics
    plt.title(f"TSP Graph {graph_index} - Tour Prediction Results\n"
              f"Predicted Edges: {total_predicted_as_1}, "
              f"Correct Solution Edges: {correct_solution_edges}/{total_actual_solution}, "
              f"Missed: {missed_solution_edges}, "
              f"Solution Accuracy: {solution_accuracy:.4f}", 
              fontsize=12)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='lightgrey', lw=2, label='Predicted Edge (Sparsified Graph)'),
        Line2D([0], [0], color='green', lw=2, label='Correct Solution Edge'),
        Line2D([0], [0], color='red', lw=2, linestyle='dashed', label='Missed Solution Edge')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Remove axis
    plt.axis('off')
    
    # Save visualization
    file_path = os.path.join(args.output_dir, f"tsp_graph_{graph_index}_tour_prediction.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {file_path}")
    
    # Show the plot
    plt.show()
    
    return solution_accuracy, correct_solution_edges, missed_solution_edges

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
        # Check if edge_labels is a tensor or numpy array and handle accordingly
        if isinstance(edge_labels[i], torch.Tensor):
            label = edge_labels[i].cpu().item()
        else:
            # Handle numpy or other types directly
            label = edge_labels[i]
            
        if label == 1:
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

def create_side_by_side_visualization(graph, edge_labels, predictions, graph_index, args):
    """Create a side-by-side visualization of original graph and prediction results."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert DGL graph to NetworkX for visualization
    g_nx = graph.to_networkx(node_attrs=['feat'], edge_attrs=['feat'])
    
    # Get node positions from node features (x,y coordinates)
    pos = {}
    for node_id, node_data in g_nx.nodes(data=True):
        pos[node_id] = node_data['feat'].cpu().numpy()
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    fig.suptitle(f"TSP Graph {graph_index} - Original vs Prediction", fontsize=16)
    
    # Get original edge order to match predictions with edges
    edge_list = list(g_nx.edges())
    
    # Prepare edge categories for prediction visualization
    edges_predicted_as_1 = []  # All edges predicted as 1 (sparsified graph)
    edges_correct_solution = []  # Correctly predicted edges that are part of the solution (label=1, pred=1)
    edges_missed_solution = []  # Edges that should be in solution but were missed (label=1, pred=0)
    
    # Get tour edges for original visualization
    tour_edges = []
    
    print(f"very very important {len(edge_list)}")
    for i, (u, v) in enumerate(edge_list):
        # Convert to CPU tensors
        pred = predictions[i].cpu().item()
        
        # Check if edge_labels is a tensor or numpy array and handle accordingly
        if isinstance(edge_labels[i], torch.Tensor):
            label = edge_labels[i].cpu().item()
        else:
            # Handle numpy or other types directly
            label = edge_labels[i]
        
        # For original graph - identify tour edges
        if label == 1:
            tour_edges.append((u, v))
        
        # For prediction graph - categorize edges
        if pred == 1:
            edges_predicted_as_1.append((u, v))
            if label == 1:
                edges_correct_solution.append((u, v))
        elif label == 1 and pred == 0:
            edges_missed_solution.append((u, v))
    
    # Calculate metrics
    total_predicted_as_1 = len(edges_predicted_as_1)
    total_actual_solution = len(tour_edges)
    correct_solution_edges = len(edges_correct_solution)
    missed_solution_edges = len(edges_missed_solution)
    solution_accuracy = correct_solution_edges / total_actual_solution if total_actual_solution > 0 else 0
    
    # ORIGINAL GRAPH (LEFT SUBPLOT)
    # Draw nodes
    nx.draw_networkx_nodes(g_nx, pos, node_size=300, node_color='lightblue', alpha=0.8, ax=ax1)
    
    # Draw all edges in light grey
    nx.draw_networkx_edges(g_nx, pos, width=1.0, alpha=0.3, edge_color='lightgrey', ax=ax1)
    
    # Draw tour edges in blue
    nx.draw_networkx_edges(g_nx, pos, edgelist=tour_edges, width=2.0, alpha=0.9, edge_color='blue', ax=ax1)
    
    # Set title for original graph
    ax1.set_title(f"Original Graph with TSP Tour\nTour Length: {total_actual_solution} edges", fontsize=12)
    ax1.axis('off')
    
    # PREDICTION GRAPH (RIGHT SUBPLOT)
    # Draw nodes
    nx.draw_networkx_nodes(g_nx, pos, node_size=300, node_color='lightblue', alpha=0.8, ax=ax2)
    
    # Draw edges predicted as 1 in light grey (sparsified graph)
    nx.draw_networkx_edges(g_nx, pos, edgelist=edges_predicted_as_1, width=1.5, alpha=0.5, edge_color='lightgrey', ax=ax2)
    
    # Draw correctly predicted solution edges in green
    nx.draw_networkx_edges(g_nx, pos, edgelist=edges_correct_solution, width=2.0, alpha=0.9, edge_color='green', ax=ax2)
    
    # Draw missed solution edges in red
    nx.draw_networkx_edges(g_nx, pos, edgelist=edges_missed_solution, width=2.0, alpha=0.9, edge_color='red', style='dashed', ax=ax2)
    
    # Set title for prediction graph
    ax2.set_title(f"Prediction Results\nPredicted Edges: {total_predicted_as_1}, Correct: {correct_solution_edges}/{total_actual_solution}, Missed: {missed_solution_edges}\nSolution Accuracy: {solution_accuracy:.4f}", fontsize=12)
    ax2.axis('off')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Actual Tour Edge'),
        Line2D([0], [0], color='lightgrey', lw=2, label='Predicted Edge'),
        Line2D([0], [0], color='green', lw=2, label='Correct Prediction'),
        Line2D([0], [0], color='red', lw=2, linestyle='dashed', label='Missed Edge')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=12, bbox_to_anchor=(0.5, 0.02))
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save visualization
    file_path = os.path.join(args.output_dir, f"tsp_graph_{graph_index}_side_by_side.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Side-by-side visualization saved to {file_path}")
    
    # Show the plot
    plt.show()
    
    return solution_accuracy, correct_solution_edges, missed_solution_edges

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
    
    # Create side-by-side visualization
    accuracy, correct_edges, missed_edges = create_side_by_side_visualization(
        graph, edge_labels, predictions, graph_index, args)
    
    print(f"Graph {graph_index} prediction accuracy: {accuracy:.4f}")
    print(f"Correct edges: {correct_edges}, Missed edges: {missed_edges}")

if __name__ == "__main__":
    main()
