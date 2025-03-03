"""
Utility functions for visualizing graphs before and after sparsification.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os

def create_color_map(probabilities=None, colormap_name='coolwarm'):
    """
    Create a color map for edge visualization based on probabilities.
    
    Parameters:
    -----------
    probabilities : list or numpy.ndarray, optional
        Edge probabilities to map to colors
    colormap_name : str, optional
        Name of the matplotlib colormap to use
        
    Returns:
    --------
    list
        List of RGB color tuples
    """
    if probabilities is None:
        return ['black'] * 100  # Default color
        
    cmap = plt.get_cmap(colormap_name)
    return [cmap(p) for p in probabilities]

def compute_layout(graph, layout_type='spring', seed=42):
    """
    Compute a layout for the graph.
    
    Parameters:
    -----------
    graph : networkx.Graph
        The graph to layout
    layout_type : str, optional
        The type of layout to use
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Node positions
    """
    if layout_type == 'spring':
        return nx.spring_layout(graph, seed=seed)
    elif layout_type == 'circular':
        return nx.circular_layout(graph)
    elif layout_type == 'kamada_kawai':
        return nx.kamada_kawai_layout(graph)
    elif layout_type == 'spectral':
        return nx.spectral_layout(graph)
    elif layout_type == 'shell':
        return nx.shell_layout(graph)
    else:
        return nx.spring_layout(graph, seed=seed)  # Default to spring

def visualize_graph_comparison(original_graph, sparsified_graph, 
                              output_path='graph_comparison.png', 
                              layout_type='spring',
                              figsize=(18, 7),  # Increased width for colorbar space
                              dpi=300,
                              node_size=40,
                              edge_width=1.0,
                              show_edges_removed=True,
                              use_same_layout=True,
                              title_original="Original Graph",
                              title_sparsified="Sparsified Graph",
                              save=True,
                              show=False):
    """
    Visualize original and sparsified graphs side by side.
    
    Parameters:
    -----------
    original_graph : networkx.Graph
        The original input graph
    sparsified_graph : networkx.Graph
        The sparsified output graph
    output_path : str, optional
        Path to save the visualization
    layout_type : str, optional
        The type of layout to use
    figsize : tuple, optional
        Figure size (width, height) in inches
    dpi : int, optional
        Resolution of the output image
    node_size : int, optional
        Size of nodes in the visualization
    edge_width : float, optional
        Width of edges in the visualization
    show_edges_removed : bool, optional
        Whether to show removed edges in light gray
    use_same_layout : bool, optional
        Whether to use the same layout for both graphs
    title_original : str, optional
        Title for the original graph subplot
    title_sparsified : str, optional
        Title for the sparsified graph subplot
    save : bool, optional
        Whether to save the visualization
    show : bool, optional
        Whether to display the visualization
        
    Returns:
    --------
    tuple
        Figure and axes objects
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Get node positions - use same layout for both graphs if specified
    if use_same_layout:
        # Calculate positions based on the union of both graphs
        combined_graph = nx.compose(original_graph, sparsified_graph)
        pos = compute_layout(combined_graph, layout_type=layout_type)
    else:
        # Calculate separate positions for each graph
        pos_original = compute_layout(original_graph, layout_type=layout_type)
        pos_sparsified = compute_layout(sparsified_graph, layout_type=layout_type)
        pos = pos_original  # Will be overridden for the second plot if use_same_layout is False
    
    # Graph statistics
    n_nodes_original = original_graph.number_of_nodes()
    n_edges_original = original_graph.number_of_edges()
    n_nodes_sparsified = sparsified_graph.number_of_nodes()
    n_edges_sparsified = sparsified_graph.number_of_edges()
    
    # First subplot: original graph
    ax = axes[0]
    ax.set_title(f"{title_original}\n({n_nodes_original} nodes, {n_edges_original} edges)")
    
    # Extract edge probabilities if they exist
    edge_colors_original = 'black'
    edge_widths_original = [edge_width] * original_graph.number_of_edges()
    
    # Draw original graph
    nx.draw_networkx(
        original_graph, pos=pos, ax=ax,
        node_size=node_size, 
        node_color='lightblue',
        edge_color=edge_colors_original,
        width=edge_widths_original,
        arrows=True if original_graph.is_directed() else False,
        with_labels=False,
        alpha=0.8
    )
    
    # Second subplot: sparsified graph
    if not use_same_layout and 'pos_sparsified' in locals():
        pos = pos_sparsified
    
    ax = axes[1]
    ax.set_title(f"{title_sparsified}\n({n_nodes_sparsified} nodes, {n_edges_sparsified} edges)")
    
    # Draw removed edges first if requested
    if show_edges_removed:
        removed_edges = set(original_graph.edges()) - set(sparsified_graph.edges())
        if removed_edges:
            removed_graph = original_graph.edge_subgraph(removed_edges)
            nx.draw_networkx_edges(
                removed_graph, pos=pos, ax=ax,
                edge_color='lightgray',
                width=0.5,
                alpha=0.3,
                arrows=True if original_graph.is_directed() else False
            )
    
    # Extract edge probabilities from sparsified graph if they exist
    edge_colors_sparsified = 'red'
    edge_widths_sparsified = [edge_width] * sparsified_graph.number_of_edges()
    
    # If probabilities are present in edge attributes, use them for colors
    if sparsified_graph.number_of_edges() > 0:
        first_edge = list(sparsified_graph.edges(data=True))[0]
        if 'pred' in first_edge[2]:
            probabilities = [data['pred'] for _, _, data in sparsified_graph.edges(data=True)]
            edge_colors_sparsified = create_color_map(probabilities)
            # Scale edge width by probability
            edge_widths_sparsified = [1.0 + 2.0 * prob for prob in probabilities]
    
    # Draw sparsified graph
    nx.draw_networkx(
        sparsified_graph, pos=pos, ax=ax,
        node_size=node_size,
        node_color='lightblue',
        edge_color=edge_colors_sparsified,
        width=edge_widths_sparsified,
        arrows=True if sparsified_graph.is_directed() else False,
        with_labels=False,
        alpha=0.8
    )
    
    # Add retention percentage in the title
    retention_pct = 100.0 * n_edges_sparsified / n_edges_original if n_edges_original > 0 else 0
    axes[1].set_title(f"{title_sparsified}\n({n_nodes_sparsified} nodes, {n_edges_sparsified} edges, {retention_pct:.1f}% retained)")
    
    # Add a colorbar if we have probability data
    if isinstance(edge_colors_sparsified, list) and len(edge_colors_sparsified) > 0:
        # Create a separate axis for the colorbar to avoid overlap with the graph
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('coolwarm'))
        sm.set_array([0, 1])
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Edge Probability')
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make room for colorbar
    
    # Save the figure if requested
    if save:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    # Show the figure if requested
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, axes

def visualize_degree_distribution(original_graph, sparsified_graph, output_path='degree_distribution.png'):
    """
    Visualize the degree distribution of original and sparsified graphs.
    
    Parameters:
    -----------
    original_graph : networkx.Graph
        The original input graph
    sparsified_graph : networkx.Graph
        The sparsified output graph
    output_path : str, optional
        Path to save the visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # In-degree distribution
    ax = axes[0]
    in_degrees_original = [d for _, d in original_graph.in_degree()] if original_graph.is_directed() else [d for _, d in original_graph.degree()]
    in_degrees_sparsified = [d for _, d in sparsified_graph.in_degree()] if sparsified_graph.is_directed() else [d for _, d in sparsified_graph.degree()]
    
    max_degree = max(max(in_degrees_original) if in_degrees_original else 0, 
                     max(in_degrees_sparsified) if in_degrees_sparsified else 0)
    bins = range(0, max_degree + 2)
    
    ax.hist([in_degrees_original, in_degrees_sparsified], bins=bins, alpha=0.7, label=['Original', 'Sparsified'])
    ax.set_title('In-Degree Distribution')
    ax.set_xlabel('In-Degree')
    ax.set_ylabel('Count')
    ax.legend()
    
    # Out-degree distribution (only for directed graphs)
    ax = axes[1]
    if original_graph.is_directed():
        out_degrees_original = [d for _, d in original_graph.out_degree()]
        out_degrees_sparsified = [d for _, d in sparsified_graph.out_degree()]
        
        max_degree = max(max(out_degrees_original) if out_degrees_original else 0, 
                         max(out_degrees_sparsified) if out_degrees_sparsified else 0)
        bins = range(0, max_degree + 2)
        
        ax.hist([out_degrees_original, out_degrees_sparsified], bins=bins, alpha=0.7, label=['Original', 'Sparsified'])
        ax.set_title('Out-Degree Distribution')
        ax.set_xlabel('Out-Degree')
    else:
        ax.hist([in_degrees_original, in_degrees_sparsified], bins=bins, alpha=0.7, label=['Original', 'Sparsified'])
        ax.set_title('Degree Distribution (Log Scale)')
        ax.set_xlabel('Degree')
        ax.set_yscale('log')
    
    ax.set_ylabel('Count')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Degree distribution visualization saved to {output_path}")

def visualize_tsp_solution(original_graph, sparsified_graph, tsp_edges=None, 
                          output_path='tsp_comparison.png',
                          layout_type='spring',
                          figsize=(15, 7),
                          dpi=300,
                          node_size=40,
                          show=False):
    """
    Visualize the TSP solution and compare it with the sparsified graph.
    
    Parameters:
    -----------
    original_graph : networkx.Graph
        The original input graph
    sparsified_graph : networkx.Graph
        The sparsified output graph
    tsp_edges : set, optional
        Set of edges in the TSP solution (will be computed if not provided)
    output_path : str, optional
        Path to save the visualization
    layout_type : str, optional
        The type of layout to use
    figsize : tuple, optional
        Figure size (width, height) in inches
    dpi : int, optional
        Resolution of the output image
    node_size : int, optional
        Size of nodes in the visualization
    show : bool, optional
        Whether to display the visualization
        
    Returns:
    --------
    tuple
        Figure and axes objects, and TSP comparison statistics
    """
    # Import TSP utilities only when needed
    try:
        from tsp_utils import compute_tsp_solution, compare_sparsified_with_tsp
        tsp_utils_available = True
    except ImportError:
        print("TSP utilities not available. Please install the required packages.")
        tsp_utils_available = False
        return None, None, None
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Compute TSP solution if not provided
    if tsp_edges is None and tsp_utils_available:
        _, _, tsp_edges = compute_tsp_solution(original_graph)
    
    # Get comparison statistics
    if tsp_utils_available:
        tsp_stats = compare_sparsified_with_tsp(original_graph, sparsified_graph, tsp_edges)
    else:
        tsp_stats = {"tsp_coverage_percent": 0, "tsp_efficiency_percent": 0}
    
    # Get node positions - use same layout for both visualizations
    combined_graph = nx.compose(original_graph, sparsified_graph)
    pos = compute_layout(combined_graph, layout_type=layout_type)
    
    # Get sets of edges
    original_edges = set(original_graph.edges())
    sparsified_edges = set(sparsified_graph.edges())
    
    # Calculate edge sets for visualization
    if tsp_edges:
        tsp_edges_not_in_sparsified = tsp_edges - sparsified_edges  # TSP edges missed by sparsification
        tsp_edges_in_sparsified = tsp_edges.intersection(sparsified_edges)  # TSP edges captured by sparsification
        non_tsp_edges_in_sparsified = sparsified_edges - tsp_edges  # Sparsified edges that are not in TSP
        non_tsp_edges_in_original = original_edges - tsp_edges  # Original edges not in TSP
    else:
        tsp_edges = set()
        tsp_edges_not_in_sparsified = set()
        tsp_edges_in_sparsified = set()
        non_tsp_edges_in_sparsified = sparsified_edges
        non_tsp_edges_in_original = original_edges
    
    # First subplot: Original graph with TSP solution
    ax = axes[0]
    ax.set_title(f"Original Graph with TSP Overlay\n"
                f"({original_graph.number_of_nodes()} nodes, {original_graph.number_of_edges()} edges)")
    
    # Draw non-TSP edges in original graph (light gray)
    if non_tsp_edges_in_original:
        non_tsp_graph = original_graph.edge_subgraph(non_tsp_edges_in_original)
        nx.draw_networkx_edges(
            non_tsp_graph, pos=pos, ax=ax,
            edge_color='lightgray',
            width=0.5,
            alpha=0.3,
            arrows=True if original_graph.is_directed() else False
        )
    
    # Draw TSP edges in original graph (green)
    if tsp_edges:
        tsp_graph = original_graph.edge_subgraph(tsp_edges)
        nx.draw_networkx_edges(
            tsp_graph, pos=pos, ax=ax,
            edge_color='green',
            width=2.0,
            alpha=0.8,
            arrows=True if original_graph.is_directed() else False
        )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        original_graph, pos=pos, ax=ax,
        node_size=node_size,
        node_color='lightblue',
        alpha=0.8
    )
    
    # Second subplot: Sparsified graph with TSP overlay
    ax = axes[1]
    ax.set_title(f"Sparsified Graph with TSP Overlay\n"
               f"({sparsified_graph.number_of_nodes()} nodes, {sparsified_graph.number_of_edges()} edges, "
               f"TSP Coverage: {tsp_stats['tsp_coverage_percent']:.1f}%)")
    
    # Draw non-TSP edges in sparsified graph (light gray)
    if non_tsp_edges_in_sparsified:
        non_tsp_sparsified_graph = sparsified_graph.edge_subgraph(non_tsp_edges_in_sparsified)
        nx.draw_networkx_edges(
            non_tsp_sparsified_graph, pos=pos, ax=ax,
            edge_color='lightgray',
            width=0.5,
            alpha=0.5,
            arrows=True if sparsified_graph.is_directed() else False
        )
    
    # Draw TSP edges that are in sparsified graph (green)
    if tsp_edges_in_sparsified:
        tsp_in_sparsified_graph = sparsified_graph.edge_subgraph(tsp_edges_in_sparsified)
        nx.draw_networkx_edges(
            tsp_in_sparsified_graph, pos=pos, ax=ax,
            edge_color='green',
            width=2.0,
            alpha=0.8,
            arrows=True if sparsified_graph.is_directed() else False
        )
    
    # Draw TSP edges that were missed by sparsification (red)
    if tsp_edges_not_in_sparsified:
        # Create a temporary graph with these edges
        missed_tsp_graph = nx.DiGraph() if original_graph.is_directed() else nx.Graph()
        missed_tsp_graph.add_nodes_from(original_graph.nodes())
        missed_tsp_graph.add_edges_from(tsp_edges_not_in_sparsified)
        
        nx.draw_networkx_edges(
            missed_tsp_graph, pos=pos, ax=ax,
            edge_color='red',
            width=2.0,
            alpha=0.8,
            style='dashed',
            arrows=True if original_graph.is_directed() else False
        )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        sparsified_graph, pos=pos, ax=ax,
        node_size=node_size,
        node_color='lightblue',
        alpha=0.8
    )
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='TSP Edge'),
        Line2D([0], [0], color='red', lw=2, linestyle='dashed', label='Missed TSP Edge'),
        Line2D([0], [0], color='lightgray', lw=1, alpha=0.5, label='Non-TSP Edge')
    ]
    axes[1].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    # Add efficiency info
    fig.text(0.5, 0.01, 
             f"TSP Efficiency: {tsp_stats['tsp_efficiency_percent']:.1f}% of sparsified edges are in TSP solution",
             ha='center', fontsize=12)
    
    plt.tight_layout()
    
    # Save the figure
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"TSP visualization saved to {output_path}")
    
    # Show the figure if requested
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, axes, tsp_stats

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize graph comparison')
    parser.add_argument('--original', type=str, required=True, help='Path to original graph file')
    parser.add_argument('--sparsified', type=str, required=True, help='Path to sparsified graph file')
    parser.add_argument('--output', type=str, default='graph_comparison.png', help='Output path for visualization')
    parser.add_argument('--layout', type=str, default='spring', choices=['spring', 'circular', 'kamada_kawai', 'spectral', 'shell'], help='Layout type')
    args = parser.parse_args()
    
    # Load graphs
    original_graph = nx.read_gpickle(args.original)
    sparsified_graph = nx.read_gpickle(args.sparsified)
    
    # Visualize
    visualize_graph_comparison(
        original_graph, 
        sparsified_graph, 
        output_path=args.output,
        layout_type=args.layout,
        show=True
    )
    
    # Visualize degree distribution
    visualize_degree_distribution(
        original_graph, 
        sparsified_graph, 
        output_path=f"{os.path.splitext(args.output)[0]}_degree_dist.png"
    )
