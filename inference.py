"""
    Script for sparsifying graphs using a trained GNN model.
    Loads a trained model and applies it to a new graph for edge classification.
"""

import os
import torch
import pickle
import numpy as np
import networkx as nx
import dgl
from nets.TSP_edge_classification.load_net import gnn_model
import argparse
from scipy.sparse import csr_matrix
from nets.TSP_edge_classification.gated_gcn_net import GatedGCNNet

# Import TSP utilities if available
try:
    from tsp_utils import compute_tsp_solution, compare_sparsified_with_tsp
    tsp_utils_available = True
except ImportError:
    print("Warning: TSP utilities not available. Install PyConcorde for TSP analysis.")
    tsp_utils_available = False

# Import visualization module
try:
    from graph_visualization import visualize_graph_comparison, visualize_degree_distribution, visualize_tsp_solution
    visualization_available = True
except ImportError:
    print("Warning: Visualization module not available. Install matplotlib for visualization.")
    visualization_available = False

def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device

def prepare_graph(nx_graph, node_features=None, edge_features=None):
    """
    Convert a NetworkX graph to a DGL graph with proper node and edge features.
    
    Parameters:
    -----------
    nx_graph : networkx.DiGraph
        Input graph to be processed
    node_features : dict, optional
        Dictionary mapping node IDs to feature vectors (x,y coordinates)
    edge_features : dict, optional
        Dictionary mapping edge tuples (u,v) to feature vectors
        
    Returns:
    --------
    dgl.DGLGraph
        Graph ready for model processing
    """
    # Create a DGL graph from the NetworkX graph
    g = dgl.from_networkx(nx_graph, edge_attrs=['weight'] if 'weight' in next(iter(nx_graph.edges(data=True)))[2] else None)
    
    # Add node features (x,y coordinates)
    if node_features is None:
        # If no coordinates provided, check if nodes have 'pos' attribute in the graph
        if nx_graph.number_of_nodes() > 0 and 'pos' in next(iter(nx_graph.nodes(data=True)))[1]:
            # Extract coordinates from node attributes
            coords = []
            for _, data in sorted(nx_graph.nodes(data=True)):
                coords.append(data['pos'])
            node_feats = torch.FloatTensor(coords)
        else:
            # Generate random 2D coordinates if not found
            node_feats = torch.rand((nx_graph.number_of_nodes(), 2))
    else:
        # Use provided coordinates
        nodes = sorted(nx_graph.nodes())
        node_feats = torch.FloatTensor([node_features[n] for n in nodes])
    
    g.ndata['feat'] = node_feats
        
    # Add edge features
    if edge_features is None:
        # Get distances between nodes based on coordinates
        coords = node_feats.numpy()
        edges = list(nx_graph.edges())
        
        # Compute edge weights if not present
        if 'weight' not in g.edata:
            # Compute Euclidean distances between connected nodes
            weights = []
            for u, v in edges:
                dist = np.sqrt(np.sum((coords[u] - coords[v])**2))
                weights.append(dist)
            
            # Add weights to graph
            g.edata['weight'] = torch.FloatTensor(weights).view(-1, 1)
        
        # Compute global min and max weights
        weights = g.edata['weight'].view(-1).numpy()
        global_max_weight = np.max(weights)
        global_min_weight = max(np.min(weights), 1e-9)
        
        # Compute per-node max and min weights
        node_max_weights = {}
        node_min_weights = {}
        
        for i, (u, v) in enumerate(edges):
            w = weights[i]
            # Update node max
            node_max_weights[u] = max(node_max_weights.get(u, 0), w)
            # Update node min
            if u not in node_min_weights:
                node_min_weights[u] = w
            else:
                node_min_weights[u] = min(node_min_weights[u], w)
        
        # Create edge features based on TSP edge format
        edge_feats = []
        for i, (u, v) in enumerate(edges):
            weight = weights[i]
            max_weight = max(node_max_weights.get(u, weight), 1e-9)
            min_weight = max(node_min_weights.get(u, weight), 1e-9)
            
            edge_feats.append([
                weight / global_max_weight,
                weight / max_weight,
                min_weight / max(weight, 1e-9),
                global_min_weight / max(weight, 1e-9)
            ])
        
        g.edata['feat'] = torch.FloatTensor(edge_feats)
    else:
        # Use provided edge features
        edges = list(nx_graph.edges())
        edge_feats = torch.FloatTensor([edge_features[e] for e in edges])
        g.edata['feat'] = edge_feats
        
    return g

def load_model(model_path, info_path, device):
    """
    Load the trained model and its associated information.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model file
    info_path : str
        Path to the model info file
    device : torch.device
        Device to load the model on
        
    Returns:
    --------
    tuple
        (model, model_info)
    """
    try:
        # Load model info
        with open(info_path, 'rb') as f:
            model_info = pickle.load(f)
        
        # Print model type if available, otherwise use a default message
        if 'net_params' in model_info:
            if 'model' in model_info['net_params']:
                print(f"Loading model {model_info['net_params']['model']}")
            else:
                print("Loading model GatedGCN (inferred from available parameters)")
        else:
            print("Loading model GatedGCN (default)")
        
        # Initialize model with parameters from info
        net_params = model_info.get('net_params', {})
        
        # Ensure critical parameters are set with defaults if missing
        if 'L' not in net_params:
            net_params['L'] = 4  # Default number of layers
        if 'hidden_dim' not in net_params:
            net_params['hidden_dim'] = 70  # Default hidden dimension
        if 'out_dim' not in net_params:
            net_params['out_dim'] = 70  # Default output dimension
        if 'in_dim' not in net_params:
            net_params['in_dim'] = 2  # Default input dimension for node features
        if 'in_dim_edge' not in net_params:
            net_params['in_dim_edge'] = 4  # Update default to 4 based on error
        if 'batch_norm' not in net_params:
            net_params['batch_norm'] = True  # Default batch norm setting
        if 'residual' not in net_params:
            net_params['residual'] = True  # Default residual connection setting
        if 'edge_feat' not in net_params:
            net_params['edge_feat'] = True  # Default edge feature setting
        if 'n_classes' not in net_params:
            net_params['n_classes'] = 1  # Binary classification for edges
        
        # Create the model
        model = GatedGCNNet(net_params)
        
        # Load state dictionary with appropriate device mapping
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Print metrics if available
        if 'best_val_metric' in model_info:
            print(f"Best validation F1: {model_info['best_val_metric']}")
        if 'test_metric' in model_info:
            print(f"Test F1: {model_info['test_metric']}")
        
        return model, model_info
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Loading GatedGCN with default parameters")
        
        # Define default parameters
        default_params = {
            'L': 4,
            'hidden_dim': 70,
            'out_dim': 70,
            'in_dim': 2,
            'in_dim_edge': 4,  # Update default to 4 based on error
            'batch_norm': True,
            'residual': True,
            'edge_feat': True,
            'n_classes': 1
        }
        
        # Create model with default parameters
        model = GatedGCNNet(default_params)
        
        # Load state dictionary if model path exists
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        
        model.to(device)
        model.eval()
        
        return model, {'net_params': default_params}

def convert_to_dgl_graph(nx_graph):
    """
    Convert a NetworkX graph to a DGL graph with appropriate node and edge features for TSP.
    
    Parameters:
    -----------
    nx_graph : networkx.Graph
        Input NetworkX graph to convert
        
    Returns:
    --------
    dgl.DGLGraph
        The converted DGL graph with node and edge features
    """
    # Check for edge attributes
    has_weight = False
    if nx_graph.number_of_edges() > 0:
        first_edge = list(nx_graph.edges(data=True))[0]
        has_weight = 'weight' in first_edge[2]
    
    # Create DGL graph from networkx
    dgl_graph = dgl.from_networkx(nx_graph, edge_attrs=['weight'] if has_weight else None)
    
    # Get node coordinates (either use existing or generate random)
    n_nodes = dgl_graph.num_nodes()
    
    # Check if nodes have position attributes
    node_coords = None
    if nx_graph.number_of_nodes() > 0:
        if 'pos' in next(iter(nx_graph.nodes(data=True)))[1]:
            # Extract positions from node attributes
            node_coords = np.array([data['pos'] for _, data in sorted(nx_graph.nodes(data=True))])
    
    # If no coordinates found, generate random ones
    if node_coords is None:
        node_coords = np.random.rand(n_nodes, 2)  # Random 2D coordinates
    
    # Set node features to coordinates
    dgl_graph.ndata['feat'] = torch.FloatTensor(node_coords)
    
    # Calculate edge features based on node coordinates and TSP requirements
    edges = list(nx_graph.edges())
    weights = []
    
    # Compute edge weights if not already present
    if not has_weight or 'weight' not in dgl_graph.edata:
        for u, v in edges:
            dist = np.sqrt(np.sum((node_coords[u] - node_coords[v])**2))
            weights.append(dist)
        
        edge_weights = np.array(weights)
    else:
        edge_weights = dgl_graph.edata['weight'].numpy().flatten()
    
    # Compute global stats
    global_max_weight = max(np.max(edge_weights), 1e-9)
    global_min_weight = max(np.min(edge_weights), 1e-9)
    
    # Compute per-node max and min weights
    node_max_weights = {}
    node_min_weights = {}
    
    for i, (u, v) in enumerate(edges):
        w = edge_weights[i]
        # Update max
        node_max_weights[u] = max(node_max_weights.get(u, 0), w)
        # Update min
        if u not in node_min_weights:
            node_min_weights[u] = w
        else:
            node_min_weights[u] = min(node_min_weights[u], w)
    
    # Create edge features according to the expected format
    edge_feats = []
    for i, (u, v) in enumerate(edges):
        weight = edge_weights[i]
        max_weight = max(node_max_weights.get(u, weight), 1e-9)  # Use node's max weight
        min_weight = max(node_min_weights.get(u, weight), 1e-9)  # Use node's min weight
        
        edge_feats.append([
            weight / global_max_weight,
            weight / max_weight,
            min_weight / max(weight, 1e-9),
            global_min_weight / max(weight, 1e-9)
        ])
    
    # Set the edge features
    dgl_graph.edata['feat'] = torch.FloatTensor(edge_feats)
    
    return dgl_graph

def apply_model(model, g, device, threshold=0.5):
    """
    Apply the trained model to a graph and return the sparsified graph.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The trained GNN model
    g : dgl.DGLGraph
        Input graph
    device : torch.device
        Device to run inference on
    threshold : float
        Probability threshold for keeping an edge
        
    Returns:
    --------
    tuple
        Original DGL graph with predictions, sparsified NetworkX graph
    """
    model.eval()
    g = g.to(device)
    
    # Get node and edge features
    h = g.ndata['feat'].to(device)
    e = g.edata['feat'].to(device)
    
    with torch.no_grad():
        # Forward pass to get edge predictions
        # The error shows a shape mismatch, possibly due to batch processing in the model
        # Let's examine the output more carefully
        model_output = model.forward(g, h, e)
        
        # Debug output shape
        # print(f"Model output shape: {[x.shape if isinstance(x, torch.Tensor) else type(x) for x in model_output]}")
        
        # Extract edge predictions - based on your model's output format
        # Common GatedGCN implementations return edge scores as one of the outputs
        if isinstance(model_output, tuple) and len(model_output) > 0:
            edge_logits = model_output[0]
            
            # Check shape of edge_logits
            # print(f"Edge logits shape: {edge_logits.shape}")
            
            # Ensure edge_logits matches the number of edges
            if edge_logits.shape[0] != g.num_edges():
                print(f"WARNING: Edge logits shape ({edge_logits.shape[0]}) doesn't match number of edges ({g.num_edges()})")
                # Try to reshape or resize if needed
                if len(edge_logits.shape) > 1 and edge_logits.shape[1] == g.num_edges():
                    # Transpose if dimensions are swapped
                    edge_logits = edge_logits.t()
                    print(f"Transposed shape: {edge_logits.shape}")
            
            # Apply sigmoid to get probabilities
            edge_probs = torch.sigmoid(edge_logits)
            
            # Ensure we have the right shape for assignments
            if len(edge_probs.shape) > 1 and edge_probs.shape[0] == g.num_edges():
                # We have the right shape
                pass
            elif len(edge_probs.shape) > 1:
                # Try to reshape to match number of edges
                edge_probs = edge_probs.reshape(g.num_edges(), -1)
            else:
                # Ensure it's 2D for assignment
                edge_probs = edge_probs.view(-1, 1)
        else:
            # Fallback to a default prediction if output format is unexpected
            print("WARNING: Unexpected model output format, using default predictions")
            edge_probs = torch.sigmoid(torch.rand(g.num_edges(), 1, device=device))
    
        # Create a new tensor for binary decisions based on threshold
        keep_mask = (edge_probs >= threshold).float()
        
        # Ensure edge_probs and keep_mask have the right shape for assignment
        print(f"Final edge_probs shape: {edge_probs.shape}, keep_mask shape: {keep_mask.shape}")
        
        # Store scalar values if the output is multi-dimensional
        if len(edge_probs.shape) > 1 and edge_probs.shape[1] > 1:
            # If we have multiple outputs per edge, take the first one or average
            prob_scalar = edge_probs[:, 0]  # Take first column
            keep_scalar = keep_mask[:, 0]   # Take first column
            g.edata['pred'] = prob_scalar.view(-1, 1)
            g.edata['keep'] = keep_scalar.view(-1, 1)
        else:
            # Otherwise use as is
            g.edata['pred'] = edge_probs
            g.edata['keep'] = keep_mask
    
    # Convert back to NetworkX for the sparsified graph
    nx_graph = dgl.to_networkx(g, edge_attrs=['pred', 'keep'])
    
    # Create sparsified graph by keeping only edges with 'keep' = 1
    sparsified_graph = nx.DiGraph()
    sparsified_graph.add_nodes_from(nx_graph.nodes(data=True))
    
    for u, v, data in nx_graph.edges(data=True):
        if data['keep'] > 0.5:  # Keep edges that are predicted to be important
            edge_data = {k: v for k, v in data.items() if k != 'keep'}
            sparsified_graph.add_edge(u, v, **edge_data)
    
    return g, sparsified_graph

def main():
    parser = argparse.ArgumentParser(description='Graph sparsification inference')
    parser.add_argument('--model_path', type=str, default='best_model.pkl', help='Path to trained model')
    parser.add_argument('--info_path', type=str, default='best_model_info.pkl', help='Path to model info')
    parser.add_argument('--use_last_epoch', action='store_true', help='Use last epoch model instead of best model')
    parser.add_argument('--graph_path', type=str, required=True, help='Path to input graph file')
    parser.add_argument('--output_path', type=str, default='sparsified_graph.gpickle', help='Output path')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for keeping edges')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID (-1 for CPU)')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--layout', type=str, default='spring', 
                       choices=['spring', 'circular', 'kamada_kawai', 'spectral', 'shell'], 
                       help='Graph layout for visualization')
    parser.add_argument('--visualize-tsp', action='store_true', help='Generate TSP solution visualizations')
    args = parser.parse_args()
    
    # Override model path if using last epoch model
    if args.use_last_epoch:
        if os.path.exists('last_model.pkl'):
            args.model_path = 'last_model.pkl'
            args.info_path = 'last_model_info.pkl'
            print("Using last epoch model instead of best model")
        else:
            print("Warning: Last epoch model not found, using specified model instead")
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    
    # Load model
    model, model_info = load_model(args.model_path, args.info_path, device)
    
    # Load input graph
    print(f"Loading input graph from {args.graph_path}")
    input_graph = nx.read_gpickle(args.graph_path)
    print(f"Input graph has {input_graph.number_of_nodes()} nodes and {input_graph.number_of_edges()} edges")
    
    # Convert to DGL
    dgl_graph = convert_to_dgl_graph(input_graph)
    
    # Apply model
    print(f"Running inference with threshold {args.threshold}")
    result_dgl_graph, sparsified_graph = apply_model(model, dgl_graph, device, args.threshold)
    
    # Save output
    print(f"Sparsified graph has {sparsified_graph.number_of_nodes()} nodes and {sparsified_graph.number_of_edges()} edges")
    print(f"Saving to {args.output_path}")
    nx.write_gpickle(sparsified_graph, args.output_path)
    
    # Generate visualizations if requested
    if args.visualize and visualization_available:
        # Create paths for visualizations
        base_path = os.path.splitext(args.output_path)[0]
        vis_path = f"{base_path}_visualization.png"
        degree_path = f"{base_path}_degree_dist.png"
        
        print("Generating graph visualizations...")
        visualize_graph_comparison(
            input_graph, 
            sparsified_graph, 
            output_path=vis_path,
            layout_type=args.layout,
            title_original="Original Graph",
            title_sparsified=f"Sparsified Graph (threshold={args.threshold})",
            show=False
        )
        
        print("Generating degree distribution comparison...")
        visualize_degree_distribution(
            input_graph, 
            sparsified_graph, 
            output_path=degree_path
        )
        
        print(f"Visualizations saved to {vis_path} and {degree_path}")
        
        # Generate TSP visualization if requested
        if args.visualize_tsp and tsp_utils_available:
            tsp_path = f"{base_path}_tsp_visualization.png"
            print("Generating TSP solution visualization...")
            
            # Compute TSP solution
            print("Computing TSP solution...")
            _, _, tsp_edges = compute_tsp_solution(input_graph)
            
            # Generate TSP visualization
            _, _, tsp_stats = visualize_tsp_solution(
                input_graph,
                sparsified_graph,
                tsp_edges=tsp_edges,
                output_path=tsp_path,
                layout_type=args.layout,
                show=False
            )
            
            # Print TSP statistics
            print(f"TSP Coverage: {tsp_stats['tsp_coverage_percent']:.2f}% of TSP edges were retained")
            print(f"TSP Efficiency: {tsp_stats['tsp_efficiency_percent']:.2f}% of sparsified edges are in TSP")
            print(f"TSP visualization saved to {tsp_path}")
        
    elif args.visualize:
        print("Visualization requested but not available. Install matplotlib to enable visualization.")

if __name__ == "__main__":
    main()
