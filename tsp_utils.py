"""
Utilities for working with Traveling Salesman Problem (TSP) solutions.

This module includes functions for computing TSP solutions using PyConcorde
and comparing TSP solutions with sparsified graphs.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import warnings

# Try to import PyConcorde for TSP solutions
try:
    from concorde.tsp import TSPSolver
    CONCORDE_AVAILABLE = True
except ImportError:
    CONCORDE_AVAILABLE = False
    warnings.warn("PyConcorde not available. Install it for TSP solution calculation.")

def get_node_coords(graph):
    """
    Extract node coordinates from a graph.
    
    Parameters:
    -----------
    graph : networkx.Graph
        Graph with node coordinates stored in node attributes
        
    Returns:
    --------
    numpy.ndarray
        Array of node coordinates with shape (n_nodes, 2)
    """
    # First check if nodes have 'pos' attribute (this is the expected format for TSP graphs)
    if graph.number_of_nodes() > 0:
        if 'pos' in next(iter(graph.nodes(data=True)))[1]:
            return np.array([data['pos'] for _, data in sorted(graph.nodes(data=True))])
    
    # If no 'pos' attribute found, check if nodes have 'feat' attribute in DGL format
    if hasattr(graph, 'ndata') and 'feat' in graph.ndata:
        return graph.ndata['feat'].numpy()
    
    # As a fallback, create random coordinates
    n_nodes = graph.number_of_nodes()
    return np.random.rand(n_nodes, 2)

def compute_tsp_solution(graph):
    """
    Compute the optimal TSP solution for a graph.
    
    Parameters:
    -----------
    graph : networkx.Graph
        Input graph with node coordinates
        
    Returns:
    --------
    tuple
        (tsp_path, tsp_length, tsp_edges)
        tsp_path: List of node indices in the tour
        tsp_length: Total tour length
        tsp_edges: Set of edges (u, v) in the tour
    """
    if not CONCORDE_AVAILABLE:
        warnings.warn("PyConcorde not available. Using a simple nearest neighbor heuristic for TSP.")
        return compute_tsp_nearest_neighbor(graph)
    
    # Get node coordinates
    coords = get_node_coords(graph)
    
    # Concorde requires integer coordinates, so we scale and round
    scale_factor = 1000000
    int_coords = (coords * scale_factor).astype(int)
    
    try:
        # Create TSP solver
        solver = TSPSolver.from_data(int_coords[:, 0], int_coords[:, 1], norm="EUC_2D")
        
        # Solve TSP
        solution = solver.solve()
        
        # Extract tour
        tour = solution.tour
        tour_length = solution.optimal_value / scale_factor
    except Exception as e:
        warnings.warn(f"Error using Concorde: {str(e)}. Falling back to nearest neighbor.")
        return compute_tsp_nearest_neighbor(graph)
    
    # Create edges from tour (connecting consecutive nodes in the tour)
    tour_edges = set()
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        tour_edges.add((u, v))
        # Add reverse edge if graph is undirected
        if not graph.is_directed():
            tour_edges.add((v, u))
    
    return tour, tour_length, tour_edges

def compute_tsp_nearest_neighbor(graph):
    """
    Compute an approximate TSP solution using the nearest neighbor heuristic.
    Used as a fallback when Concorde is not available.
    
    Parameters:
    -----------
    graph : networkx.Graph
        Input graph with node coordinates
        
    Returns:
    --------
    tuple
        (tsp_path, tsp_length, tsp_edges)
        tsp_path: List of node indices in the tour
        tsp_length: Total tour length
        tsp_edges: Set of edges (u, v) in the tour
    """
    # Get node coordinates
    coords = get_node_coords(graph)
    n_nodes = coords.shape[0]
    
    # Compute distance matrix
    dist_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            dist_matrix[i, j] = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
    
    # Start from node 0
    current_node = 0
    tour = [current_node]
    unvisited = set(range(1, n_nodes))
    tour_length = 0
    
    # Build tour by selecting nearest unvisited node
    while unvisited:
        nearest_neighbor = min(unvisited, key=lambda x: dist_matrix[current_node, x])
        tour_length += dist_matrix[current_node, nearest_neighbor]
        current_node = nearest_neighbor
        tour.append(current_node)
        unvisited.remove(current_node)
    
    # Return to start
    tour_length += dist_matrix[current_node, 0]
    
    # Create edges from tour
    tour_edges = set()
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        tour_edges.add((u, v))
        # Add reverse edge if graph is undirected
        if not graph.is_directed():
            tour_edges.add((v, u))
    
    return tour, tour_length, tour_edges

def compare_sparsified_with_tsp(original_graph, sparsified_graph, tsp_edges=None):
    """
    Compare a sparsified graph with the optimal TSP solution.
    
    Parameters:
    -----------
    original_graph : networkx.Graph
        Original input graph
    sparsified_graph : networkx.Graph
        Sparsified graph
    tsp_edges : set, optional
        Set of edges in the optimal TSP solution.
        If None, it will be computed.
        
    Returns:
    --------
    dict
        Statistics about the comparison
    """
    # Compute TSP solution if not provided
    if tsp_edges is None:
        _, _, tsp_edges = compute_tsp_solution(original_graph)
    
    # Get sets of edges
    original_edges = set(original_graph.edges())
    sparsified_edges = set(sparsified_graph.edges())
    
    # Calculate metrics
    tsp_edges_count = len(tsp_edges)
    tsp_edges_in_sparsified = tsp_edges.intersection(sparsified_edges)
    tsp_edges_in_sparsified_count = len(tsp_edges_in_sparsified)
    
    # Calculate percentages
    tsp_coverage = 100 * tsp_edges_in_sparsified_count / tsp_edges_count if tsp_edges_count > 0 else 0
    sparsification_ratio = 100 * len(sparsified_edges) / len(original_edges) if len(original_edges) > 0 else 0
    
    # Calculate efficiency (percentage of sparsified edges that are in TSP)
    tsp_efficiency = 100 * tsp_edges_in_sparsified_count / len(sparsified_edges) if len(sparsified_edges) > 0 else 0
    
    return {
        "tsp_edges_total": tsp_edges_count,
        "tsp_edges_in_sparsified": tsp_edges_in_sparsified_count,
        "tsp_coverage_percent": tsp_coverage,
        "sparsification_ratio_percent": sparsification_ratio,
        "tsp_efficiency_percent": tsp_efficiency,
        "sparsified_edges_total": len(sparsified_edges),
        "original_edges_total": len(original_edges),
        "tsp_edges_set": tsp_edges,
        "tsp_edges_in_sparsified_set": tsp_edges_in_sparsified,
        "tsp_edges_missed_set": tsp_edges - sparsified_edges
    }

# If running this script directly, test with a small example
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create a small test graph
    G = nx.DiGraph()
    n_nodes = 20
    
    # Add nodes with 2D coordinates
    for i in range(n_nodes):
        # Random coordinates in [0,1] x [0,1]
        G.add_node(i, pos=np.random.rand(2))
    
    # Add edges (fully connected)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:  # No self-loops
                # Calculate Euclidean distance between nodes
                dist = np.sqrt(np.sum((G.nodes[i]['pos'] - G.nodes[j]['pos'])**2))
                G.add_edge(i, j, weight=dist)
    
    # Compute TSP solution
    tour, tour_length, tour_edges = compute_tsp_solution(G)
    print(f"TSP Tour: {tour}")
    print(f"TSP Length: {tour_length:.2f}")
    print(f"Number of TSP edges: {len(tour_edges)}")
    
    # Visualize the tour
    plt.figure(figsize=(10, 10))
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw all edges lightly
    nx.draw_networkx_edges(G, pos, alpha=0.1)
    
    # Create a subgraph with only TSP edges
    tsp_graph = G.edge_subgraph(tour_edges)
    nx.draw_networkx_edges(tsp_graph, pos, edge_color='r', width=2)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=50)
    
    plt.title("TSP Solution")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("tsp_test.png", dpi=300)
    print("Test visualization saved to tsp_test.png")
