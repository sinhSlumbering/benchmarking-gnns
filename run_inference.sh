#!/bin/bash

# Add helpful message about generating graphs
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: ./run_inference.sh MODEL_PATH GRAPH_PATH [THRESHOLD] [--visualize] [--visualize-tsp] [--use-last-epoch]"
    echo ""
    echo "To generate a new random TSP graph for testing:"
    echo "  python generate_random_tsp_graph.py --n_nodes 100 --output random_tsp_graph.gpickle --visualize"
    exit 0
fi

# Check if model_path exists, if not show error
if [ ! -f "$1" ]; then
    echo "Model file not found: $1"
    echo "Usage: ./run_inference.sh MODEL_PATH GRAPH_PATH [THRESHOLD] [--visualize] [--visualize-tsp] [--use-last-epoch]"
    exit 1
fi

# Check if graph_path exists, if not show error
if [ ! -f "$2" ]; then
    echo "Graph file not found: $2"
    echo "Usage: ./run_inference.sh MODEL_PATH GRAPH_PATH [THRESHOLD] [--visualize] [--visualize-tsp] [--use-last-epoch]"
    exit 1
fi

MODEL_PATH=$1
GRAPH_PATH=$2
THRESHOLD=${3:-0.5}  # Default threshold is 0.5 if not specified
VIZ_FLAG=""
TSP_FLAG=""
LAST_EPOCH_FLAG=""

# Check for flags
for arg in "$@"; do
    if [ "$arg" = "--visualize" ]; then
        VIZ_FLAG="--visualize"
    elif [ "$arg" = "--visualize-tsp" ]; then
        TSP_FLAG="--visualize-tsp"
    elif [ "$arg" = "--use-last-epoch" ]; then
        LAST_EPOCH_FLAG="--use_last_epoch"
    fi
done

# If threshold was not provided but flags were
if [[ "$3" == --* ]]; then
    THRESHOLD=0.5
fi

# Check if model info file exists
INFO_PATH="${MODEL_PATH%.*}_info.pkl"
if [ ! -f "$INFO_PATH" ]; then
    echo "Model info file not found: $INFO_PATH"
    echo "Creating a default model info file..."
    python create_model_info.py --model_path "$MODEL_PATH" --output_path "$INFO_PATH"
fi

# Run inference
echo "Running inference with:"
echo "  Model: $MODEL_PATH"
echo "  Info: $INFO_PATH"
echo "  Graph: $GRAPH_PATH"
echo "  Threshold: $THRESHOLD"
if [ -n "$VIZ_FLAG" ]; then
    echo "  Visualization: Enabled"
fi
if [ -n "$TSP_FLAG" ]; then
    echo "  TSP Visualization: Enabled"
fi
if [ -n "$LAST_EPOCH_FLAG" ]; then
    echo "  Using last epoch model: Enabled"
fi

python inference.py --model_path "$MODEL_PATH" --info_path "$INFO_PATH" \
    --graph_path "$GRAPH_PATH" --threshold "$THRESHOLD" \
    $VIZ_FLAG $TSP_FLAG $LAST_EPOCH_FLAG
