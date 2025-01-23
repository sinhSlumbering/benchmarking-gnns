#!/bin/bash

# Array of node counts and corresponding training sample sizes
declare -A node_samples=(
    [100]=10000
    [200]=10000
    [500]=5000
#    [1000]=1000
#    [10000]=500
)

# Number of validation and test samples
val_test_samples=100

# Base filename
base_filename="tsp"

# Loop through the node counts
for nodes in "${!node_samples[@]}"; do
    num_samples="${node_samples[$nodes]}"

    # Create directories
    mkdir -p "${base_filename}${nodes}-${nodes}_test"
    mkdir -p "${base_filename}${nodes}-${nodes}_train"
    mkdir -p "${base_filename}${nodes}-${nodes}_val"
    #
    # Generate a random seed
    seed=$((RANDOM * RANDOM))

    # Generate training data
    python data/TSP/generate_TSP.py \
        --min_nodes "$nodes" \
        --max_nodes "$nodes" \
        --num_samples "$num_samples" \
        --filename "${base_filename}${nodes}-${nodes}_train/train.txt" \
        --node_dim 2 \
        --seed "$seed"

    # Generate validation data (using a different seed)
    val_seed=$((seed + 123)) #add 123 to make it different
    python data/TSP/generate_TSP.py \
        --min_nodes "$nodes" \
        --max_nodes "$nodes" \
        --num_samples "$val_test_samples" \
        --filename "${base_filename}${nodes}-${nodes}_val/val.txt" \
        --node_dim 2 \
        --seed "$val_seed"

    # Generate test data (using yet another different seed)
    test_seed=$((seed + 456)) #add 456 to make it different
    python data/TSP/generate_TSP.py \
        --min_nodes "$nodes" \
        --max_nodes "$nodes" \
        --num_samples "$val_test_samples" \
        --filename "${base_filename}${nodes}-${nodes}_test/test.txt" \
        --node_dim 2 \
        --seed "$test_seed"

    echo "Generated data for ${nodes} nodes."
done

echo "Finished generating all datasets."
