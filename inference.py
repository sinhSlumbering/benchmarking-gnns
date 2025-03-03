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
        Dictionary mapping node IDs to feature vectors
    edge_features : dict, optional
        Dictionary mapping edge tuples (u,v) to feature vectors
        
    Returns:
    --------
    dgl.DGLGraph
        Graph ready for model processing
    """
    # Create a DGL graph from the NetworkX graph
    g = dgl.from_networkx(nx_graph, edge_attrs=['weight'] if 'weight' in next(iter(nx_graph.edges(data=True)))[2] else None)
    
    # Add default node features if not provided
    if node_features is None:
        # Default: use node degree as feature
        in_degrees = torch.FloatTensor([nx_graph.in_degree(i) for i in range(nx_graph.number_of_nodes())])
        out_degrees = torch.FloatTensor([nx_graph.out_degree(i) for i in range(nx_graph.number_of_nodes())])
        node_feats = torch.stack([in_degrees, out_degrees], dim=1)
        g.ndata['feat'] = node_feats
    else:
        # Convert the provided node features to tensor
        nodes = sorted(nx_graph.nodes())
        node_feats = torch.FloatTensor([node_features[n] for n in nodes])
        g.ndata['feat'] = node_feats
        
    # Add default edge features if not provided
    if edge_features is None:
        # Default: use edge weight as feature (or 1.0 if no weight)
        if 'weight' in g.edata:
            edge_feats = torch.FloatTensor(g.edata['weight']).view(-1, 1)
        else:
            edge_feats = torch.ones(g.number_of_edges(), 1)
        g.edata['feat'] = edge_feats
    else:
        # Convert the provided edge features to tensor
        edges = list(nx_graph.edges())
        edge_feats = torch.FloatTensor([edge_features[e] for e in edges])
        g.edata['feat'] = edge_feats
        
    return g

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
    
    with torch.no_grad():
        # Forward pass to get edge predictions
        edge_logits = model.forward(g)[0]
        edge_probs = torch.sigmoid(edge_logits)
        
        # Store predictions in the graph
        g.edata['pred'] = edge_probs
        g.edata['keep'] = (edge_probs >= threshold).float()
    
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
    parser = argparse.ArgumentParser(description='Run inference on a graph using a trained GNN')
    parser.add_argument('--model_path', default='best_model.pkl', help='Path to the trained model')
    parser.add_argument('--info_path', default='best_model_info.pkl', help='Path to model info file')
    parser.add_argument('--graph_path', required=True, help='Path to the input graph (NetworkX format)')
    parser.add_argument('--output_path', default='sparsified_graph.gpickle', help='Path to save the output graph')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for edge classification')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID (use -1 for CPU)')
    args = parser.parse_args()
    
    # Set device
    device = gpu_setup(args.gpu >= 0, args.gpu) if args.gpu >= 0 else torch.device('cpu')
    
    # Load model info and configuration
    with open(args.info_path, 'rb') as f:
        model_info = pickle.load(f)
    
    model_name = model_info['model_name']
    net_params = model_info['net_params']
    
    print(f"Loading model {model_name}")
    print(f"Best validation F1: {model_info['best_val_f1']}")
    print(f"Test F1: {model_info['test_f1']}")
    
    # Initialize the model architecture
    model = gnn_model(model_name, net_params)
    
    # Load saved weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    
    # Load input graph
    print(f"Loading input graph from {args.graph_path}")
    if args.graph_path.endswith('.gpickle'):
        input_graph = nx.read_gpickle(args.graph_path)
    else:
        # Try to load as EdgeList if not a pickle file
        input_graph = nx.read_edgelist(args.graph_path, create_using=nx.DiGraph())
    
    print(f"Input graph has {input_graph.number_of_nodes()} nodes and {input_graph.number_of_edges()} edges")
    
    # Prepare graph for model
    dgl_graph = prepare_graph(input_graph)
    
    # Apply model for inference
    print(f"Running inference with threshold {args.threshold}")
    result_dgl_graph, sparsified_graph = apply_model(model, dgl_graph, device, args.threshold)
    
    # Print results
    print(f"Sparsified graph has {sparsified_graph.number_of_nodes()} nodes and {sparsified_graph.number_of_edges()} edges")
    print(f"Kept {sparsified_graph.number_of_edges() / input_graph.number_of_edges() * 100:.2f}% of original edges")
    
    # Save the result
    nx.write_gpickle(sparsified_graph, args.output_path)
    print(f"Sparsified graph saved to {args.output_path}")
    
    # Optionally save a visualization-friendly format
    if args.output_path.endswith('.gpickle'):
        edgelist_path = args.output_path.replace('.gpickle', '.edgelist')
        nx.write_edgelist(sparsified_graph, edgelist_path)
        print(f"Edge list format also saved to {edgelist_path}")

if __name__ == "__main__":
    main()
