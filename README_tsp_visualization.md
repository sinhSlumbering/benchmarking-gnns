# TSP Graph Visualization

This tool visualizes the results of TSP edge predictions by comparing model predictions with ground truth labels.

## Overview

The visualization script (`visualize_tsp_predictions.py`) creates two primary visualizations:
1. **Actual Tour**: Shows the complete TSP tour (ground truth)
2. **Model's Predicted Tour**: Shows only the edges that the model predicts as part of the tour (prediction=1):
   - Green edges: Correctly predicted tour edges (true positives)
   - Red edges: Falsely predicted tour edges (false positives)

## How to Use

### Basic Usage

```bash
python visualize_tsp_predictions.py
```

This will randomly select a graph from the test set and create the visualizations.

### Specific Graph

```bash
python visualize_tsp_predictions.py --graph_index 5
```

### Additional Options

```bash
# Show all visualization types (including the previous implementation)
python visualize_tsp_predictions.py --show_all_visualizations

# Custom model path and output directory
python visualize_tsp_predictions.py --model_path path/to/model.pkl --output_dir ./my_visualizations
```

## Understanding the Visualizations

### 1. Actual Tour Visualization
- Blue edges show the complete optimal TSP tour (ground truth)
- No predictions are shown, just the reference solution

### 2. Model's Predicted Tour
- Shows only edges that the model predicted as part of the tour (prediction=1)
- Green edges: Correctly predicted tour edges (edges that are actually in the tour)
- Red edges: Falsely predicted tour edges (edges that are not actually in the tour)
- This visualization shows what the model thinks the tour looks like

### 3. Optional: Tour Prediction Results (with --show_all_visualizations)
- Shows only the edges that are part of the true TSP tour
- Green edges: Tour edges correctly identified by the model
- Red edges: Tour edges missed by the model

## Metrics Displayed

The visualizations include:
- Number of edges predicted as part of the tour
- Number of actual tour edges
- Precision: Percentage of predicted tour edges that are actually in the tour
- Recall: Percentage of actual tour edges that were correctly identified

## Requirements

- PyTorch
- NetworkX
- Matplotlib
- NumPy
- The trained GNN model

## Example

After running the script, you should see two image files in the output directory:
- `tsp_graph_{index}_actual_tour.png` - The ground truth tour
- `tsp_graph_{index}_predicted_tour.png` - The model's predicted tour
