"""
Utility script to generate random graphs with TSP-compatible features.
This creates graphs with proper node coordinates and edge features suitable for 
the TSP edge classification task.
"""

import networkx as nx
import numpy as np
import pickle
import argparse
import os
import matplotlib.pyplot as plt

def generate_tsp_graph(n_nodes=100, edge_density=0.1, is_complete=False, seed=None):
    """
    Generate a random graph with TSP-compatible features.
    
    Parameters:
    -----------
    n_nodes : int
        Number of nodes
    edge_density : float
        Probability of edge creation (ignored if is_complete=True)
    is_complete : bool
        Whether to create a complete graph
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    networkx.DiGraph
        Graph with node positions and edge weights
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate a random directed graph
    if is_complete:
        G = nx.complete_graph(n_nodes, create_using=nx.DiGraph)
    else:
        G = nx.erdos_renyi_graph(n_nodes, edge_density, directed=True, seed=seed)
    
    # Generate random 2D coordinates for nodes
    coords = np.random.rand(n_nodes, 2)
    
    # Add coordinates as node attributes
    for i in range(n_nodes):
        G.nodes[i]['pos'] = coords[i]
    
    # Calculate edge weights as Euclidean distances
    for u, v in G.edges():
        dist = np.sqrt(np.sum((coords[u] - coords[v])**2))
        G[u][v]['weight'] = dist
    
    return G

def visualize_graph(G, output_path=None):
    """
    Visualize the graph with node positions.
    
    Parameters:
    -----------
    G : networkx.Graph
        Graph to visualize
    output_path : str, optional
        Path to save visualization
    """
    # Extract node positions
    pos = nx.get_node_attributes(G, 'pos')
    if not pos:
        pos = nx.spring_layout(G)
    
    plt.figure(figsize=(10, 10))
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5, arrows=True if G.is_directed() else False)
    
    # Draw edge weights for a subset of edges if there are many
    if G.number_of_edges() <= 50:
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title(f"Random TSP Graph ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Graph visualization saved to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate random TSP graph')
    parser.add_argument('--n_nodes', type=int, default=100, help='Number of nodes')
    parser.add_argument('--edge_density', type=float, default=0.1, help='Edge density')
    parser.add_argument('--complete', action='store_true', help='Create complete graph')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--output', type=str, default='random_tsp_graph.gpickle', help='Output path')
    parser.add_argument('--visualize', action='store_true', help='Visualize and save graph image')
    args = parser.parse_args()
    
    # Generate graph
    print(f"Generating {'complete' if args.complete else 'random'} TSP graph with {args.n_nodes} nodes...")
    G = generate_tsp_graph(args.n_nodes, args.edge_density, args.complete, args.seed)
    
    print(f"Graph generated with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Save graph
    nx.write_gpickle(G, args.output)
    print(f"Graph saved to {args.output}")
    
    # Visualize graph
    if args.visualize:
        vis_path = os.path.splitext(args.output)[0] + '.png'
        print("Creating visualization...")
        visualize_graph(G, vis_path)

if __name__ == '__main__':
    main()
