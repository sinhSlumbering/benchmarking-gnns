#!/bin/bash

# Infinite loop to continuously check memory usage
while true
do
    # Clear the screen
    clear

    # Get the current timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Query the memory usage of each GPU
    mem_usage=$(nvidia-smi --query-gpu=name,index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)
    
    # Print the header
    echo "$timestamp - GPU Memory Usage:"
    echo "------------------------------------------"

    # Print the memory usage for each GPU
    echo "$mem_usage" | while IFS=',' read -r name idx mem_used mem_total util_gpu
    do
        echo "GPU$idx ($name): Memory Used: ${mem_used}MiB / ${mem_total}MiB, GPU Utilization: ${util_gpu}%"
    done

    # Wait for 2 seconds before the next check
    sleep 2
done
