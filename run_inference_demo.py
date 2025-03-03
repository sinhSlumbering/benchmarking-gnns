"""
Script to generate sample graphs and run inference on them.
This demonstrates how to use the inference.py script for graph sparsification.
"""

import os
import subprocess
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
import pickle
from pathlib import Path

def create_directory(directory):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def generate_random_graph(n_nodes=50, edge_prob=0.2, seed=None):
    """Generate a random directed graph with random weights"""
    G = nx.gnp_random_graph(n=n_nodes, p=edge_prob, directed=True, seed=seed)
    
    # Add random weights
    for u, v in G.edges():
        G[u][v]['weight'] = random.uniform(0.1, 1.0)
    
    # Add some node features (x,y coordinates)
    for i in G.nodes():
        G.nodes[i]['pos'] = (random.uniform(0, 100), random.uniform(0, 100))
        G.nodes[i]['feature'] = random.uniform(0, 1)
    
    return G

def generate_grid_graph(grid_size=7):
    """Generate a grid graph with random weights"""
    G = nx.grid_2d_graph(grid_size, grid_size)
    # Convert to directed by adding edges in both directions
    G = nx.DiGraph(G)
    
    # Add random weights
    for u, v in G.edges():
        G[u][v]['weight'] = random.uniform(0.1, 1.0)
    
    # Use positions from grid layout
    pos = {(x, y): (x, y) for x, y in G.nodes()}
    nx.set_node_attributes(G, pos, 'pos')
    
    # Add node features
    for i in G.nodes():
        G.nodes[i]['feature'] = random.uniform(0, 1)
    
    return G

def generate_tsp_graph(n_cities=20, seed=None):
    """Generate a complete graph representing a TSP problem"""
    if seed:
        random.seed(seed)
        np.random.seed(seed)
    
    # Generate random city positions
    positions = {i: (random.uniform(0, 100), random.uniform(0, 100)) for i in range(n_cities)}
    
    # Create complete graph
    G = nx.complete_graph(n_cities, create_using=nx.DiGraph())
    
    # Set edge weights based on Euclidean distance
    for u, v in G.edges():
        pos_u = positions[u]
        pos_v = positions[v]
        dist = np.sqrt((pos_u[0] - pos_v[0])**2 + (pos_u[1] - pos_v[1])**2)
        G[u][v]['weight'] = dist
    
    # Set node attributes
    nx.set_node_attributes(G, positions, 'pos')
    
    return G

def save_graph(G, filename):
    """Save graph to disk"""
    nx.write_gpickle(G, filename)
    print(f"Graph saved to {filename}")

def visualize_graph(G, title, output_file=None, pos=None):
    """Visualize graph"""
    plt.figure(figsize=(10, 8))
    
    # Get positions for nodes
    if pos is None:
        if all('pos' in G.nodes[n] for n in G.nodes()):
            pos = nx.get_node_attributes(G, 'pos')
        else:
            pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color='lightblue')
    
    # Draw edges with width proportional to weight
    edge_weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
    if edge_weights:
        max_weight = max(edge_weights)
        edge_widths = [w/max_weight * 2 for w in edge_weights]
    else:
        edge_widths = [1.0]
        
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7, 
                          arrowsize=15, arrowstyle='->')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    plt.title(title)
    plt.axis('off')
    
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    else:
        plt.show()
    
    plt.close()

def run_inference(model_path, info_path, graph_path, output_path, threshold=0.5, gpu=-1):
    """Run inference.py on the given graph"""
    cmd = [
        "python", "inference.py",
        "--model_path", model_path,
        "--info_path", info_path,
        "--graph_path", graph_path,
        "--output_path", output_path,
        "--threshold", str(threshold),
        "--gpu", str(gpu)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Inference completed successfully")
        print(result.stdout)
        return True
    else:
        print("Error running inference:")
        print(result.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate graphs and run inference")
    parser.add_argument("--model_path", default="trained_models/best_model.pkl", help="Path to trained model")
    parser.add_argument("--info_path", default="trained_models/best_model_info.pkl", help="Path to model info")
    parser.add_argument("--output_dir", default="inference_results", help="Directory to save results")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for edge classification")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU ID (-1 for CPU)")
    parser.add_argument("--visualize", action="store_true", help="Visualize graphs")
    args = parser.parse_args()
    
    # Create directories
    graph_dir = os.path.join(args.output_dir, "graphs")
    results_dir = os.path.join(args.output_dir, "sparsified")
    viz_dir = os.path.join(args.output_dir, "visualizations")
    
    create_directory(graph_dir)
    create_directory(results_dir)
    create_directory(viz_dir)
    
    # Check if model exists, if not, create a dummy model for demo purposes
    if not os.path.exists(args.model_path):
        print("Model not found, creating directories for models")
        model_dir = os.path.dirname(args.model_path)
        create_directory(model_dir)
        print(f"Please put your trained model in {args.model_path} and model info in {args.info_path}")
        return
    
    # Generate and process different graph types
    graph_generators = [
        ("random", lambda: generate_random_graph(n_nodes=50, edge_prob=0.2, seed=42)),
        ("grid", lambda: generate_grid_graph(grid_size=7)),
        ("tsp", lambda: generate_tsp_graph(n_cities=20, seed=42))
    ]
    
    for graph_type, generator in graph_generators:
        print(f"\n=== Processing {graph_type} graph ===")
        
        # Generate and save graph
        G = generator()
        graph_path = os.path.join(graph_dir, f"{graph_type}_graph.gpickle")
        save_graph(G, graph_path)
        
        # Visualize original graph
        if args.visualize:
            viz_path = os.path.join(viz_dir, f"{graph_type}_original.png")
            visualize_graph(G, f"Original {graph_type} Graph", viz_path)
        
        # Run inference
        output_path = os.path.join(results_dir, f"{graph_type}_sparsified.gpickle")
        success = run_inference(
            args.model_path, 
            args.info_path,
            graph_path,
            output_path,
            args.threshold,
            args.gpu
        )
        
        # Visualize result if successful
        if success and args.visualize and os.path.exists(output_path):
            try:
                sparsified_G = nx.read_gpickle(output_path)
                viz_path = os.path.join(viz_dir, f"{graph_type}_sparsified.png")
                pos = None
                if all('pos' in G.nodes[n] for n in G.nodes()):
                    pos = nx.get_node_attributes(G, 'pos')
                visualize_graph(sparsified_G, f"Sparsified {graph_type} Graph", viz_path, pos)
                
                # Calculate sparsification stats
                original_edges = G.number_of_edges()
                sparsified_edges = sparsified_G.number_of_edges()
                reduction = (original_edges - sparsified_edges) / original_edges * 100
                print(f"Edge reduction: {reduction:.2f}% ({sparsified_edges}/{original_edges} edges kept)")
            except Exception as e:
                print(f"Error visualizing sparsified graph: {e}")

if __name__ == "__main__":
    main()
