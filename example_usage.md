# Graph Sparsification with Trained GNN

This document shows how to use the trained model for graph sparsification.

## Training the Model

First, train the model using the standard command:

```bash
python main_TSP_edge_classification.py --config configs/TSP_edge_classification_GatedGCN_100k.json
```

This will save the best model during training as `best_model.pkl` in the current directory along with its configuration in `best_model_info.pkl`.

## Running Inference

After training, you can apply the model to sparsify new graphs:

```bash
python inference.py --graph_path path/to/your/graph.gpickle --output_path sparsified_result.gpickle
```

### Arguments

- `--model_path`: Path to the trained model file (default: `best_model.pkl`)
- `--info_path`: Path to the model info file (default: `best_model_info.pkl`)
- `--graph_path`: Path to input graph file (NetworkX format, required)
- `--output_path`: Where to save the sparsified graph (default: `sparsified_graph.gpickle`)
- `--threshold`: Probability threshold for keeping edges (default: 0.5)
- `--gpu`: GPU ID to use (-1 for CPU, default: -1)

### Input Graph Format

The script supports:
- NetworkX pickle files (*.gpickle)
- Edge list files

For custom input graphs, you can create one using NetworkX:

```python
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges
G.add_nodes_from(range(20))
G.add_edges_from([(i, i+1) for i in range(19)])
G.add_edges_from([(i, i+2) for i in range(18)])  # Add some shortcuts

# Save the graph
nx.write_gpickle(G, "my_test_graph.gpickle")
```

## Example with a Random Graph

```python
import networkx as nx
import random

# Create random graph
G = nx.gnp_random_graph(100, 0.05, directed=True)

# Add some weights
for u, v in G.edges():
    G[u][v]['weight'] = random.random()

# Save the graph
nx.write_gpickle(G, "random_graph.gpickle")
```

Then sparsify it:

```bash
python inference.py --graph_path random_graph.gpickle --threshold 0.7
```
