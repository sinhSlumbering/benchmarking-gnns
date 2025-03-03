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
- `--visualize`: Generate visualizations of the original and sparsified graphs
- `--visualize-tsp`: Generate TSP solution overlay visualizations
- `--layout`: Layout algorithm for visualization (spring, circular, kamada_kawai, spectral, shell)

### Visualization

To generate visualizations of the original and sparsified graphs, add the `--visualize` flag:

```bash
python inference.py --graph_path random_graph.gpickle --threshold 0.7 --visualize
```

This will create two image files alongside your output graph:
1. `sparsified_graph_visualization.png` - Side-by-side comparison of original and sparsified graphs
2. `sparsified_graph_degree_dist.png` - Comparison of degree distributions

### TSP Solution Visualization

To analyze how well the sparsified graph preserves TSP solutions, add the `--visualize-tsp` flag:

```bash
python inference.py --graph_path random_graph.gpickle --threshold 0.7 --visualize --visualize-tsp
```

This generates an additional visualization:
- `sparsified_graph_tsp_visualization.png` - Shows TSP path overlay with:
  - Green: TSP edges that were retained in the sparsified graph
  - Red (dashed): TSP edges that were lost during sparsification
  - Light gray: Non-TSP edges

The visualization also includes statistics about:
- TSP Coverage: Percentage of TSP edges retained in sparsified graph
- TSP Efficiency: Percentage of sparsified edges that are part of the TSP solution

Note: TSP visualization requires PyConcorde. Install it with:
```bash
pip install concorde
```

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

Then sparsify it with visualization including TSP analysis:

```bash
python inference.py --graph_path random_graph.gpickle --threshold 0.7 --visualize --visualize-tsp
```

## Using the Helper Script

For convenience, you can use the `run_inference.sh` script:

```bash
./run_inference.sh best_model.pkl random_graph.gpickle 0.7 --visualize --visualize-tsp
```

This script automatically checks for and creates the model info file if needed.
