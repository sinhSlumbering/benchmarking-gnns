# Graph Sparsification with GNNs

This repository contains scripts for sparsifying graphs using trained Graph Neural Networks (GNNs). The model classifies edges in a graph as important or not, resulting in a sparsified graph that preserves the most crucial connections.

## Scripts

### inference.py

This script loads a trained GNN model and applies it to a graph for edge classification.

```bash
python inference.py --model_path <path_to_model> --info_path <path_to_model_info> --graph_path <input_graph> --output_path <output_graph> --threshold 0.5 --gpu 0
```

Arguments:
- `--model_path`: Path to the trained model file (.pkl)
- `--info_path`: Path to the model info file containing model architecture and parameters
- `--graph_path`: Path to the input graph (NetworkX format, .gpickle)
- `--output_path`: Path to save the sparsified graph
- `--threshold`: Probability threshold for keeping an edge (default: 0.5)
- `--gpu`: GPU ID to use (use -1 for CPU, default)

### run_inference_demo.py

This script generates sample graphs, runs the inference on them, and visualizes the results.

```bash
python run_inference_demo.py --model_path trained_models/best_model.pkl --info_path trained_models/best_model_info.pkl --output_dir inference_results --visualize
```

Arguments:
- `--model_path`: Path to the trained model
- `--info_path`: Path to the model info file
- `--output_dir`: Directory to save results
- `--threshold`: Classification threshold (default: 0.5)
- `--gpu`: GPU ID (-1 for CPU)
- `--visualize`: Flag to enable graph visualization

## Directory Structure

After running the demo script, the following directories will be created:

- `inference_results/graphs/`: Contains the generated input graphs
- `inference_results/sparsified/`: Contains the sparsified output graphs
- `inference_results/visualizations/`: Contains visualizations of original and sparsified graphs

## Examples

The demo script generates three types of graphs:
1. Random directed graphs
2. Grid graphs
3. TSP-like complete graphs

For each graph, it runs the inference and produces a sparsified version.

## Requirements

- Python 3.6+
- NetworkX
- PyTorch
- DGL (Deep Graph Library)
- Matplotlib
- NumPy
- SciPy
