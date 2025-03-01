import os
import json
import subprocess
from optimize_training import print_memory_stats

def create_optimized_config():
    """Create an optimized configuration file"""
    config_path = "config_templates/optimized_config.json"
    
    # Create config directory if it doesn't exist
    os.makedirs("config_templates", exist_ok=True)
    
    # Create default optimized config
    optimized_config = {
        "gpu": {"use": True, "id": 0},
        "model": "GatedGCN",
        "dataset": "TSP",
        "out_dir": "out/",
        "params": {
            "seed": 41,
            "epochs": 500,
            "batch_size": 64,
            "init_lr": 0.001,
            "lr_reduce_factor": 0.5,
            "lr_schedule_patience": 5,
            "min_lr": 1e-6,
            "weight_decay": 0.0,
            "print_epoch_interval": 5,
            "max_time": 24,
            "use_chunked_training": True,
            "chunk_size": 500,
            "epochs_per_chunk": 2,
            "num_workers": 4,
            "pin_memory": True,
            "use_amp": True
        },
        "net_params": {
            "L": 4,
            "hidden_dim": 64,
            "out_dim": 64,
            "residual": True,
            "edge_feat": True,
            "readout": "mean",
            "in_feat_dropout": 0.0,
            "dropout": 0.0,
            "batch_norm": True,
            "layer_type": "edgereprfeat"
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(optimized_config, f, indent=4)
    
    return config_path

def main():
    """Run training with optimized parameters"""
    print("Starting optimized training run")
    print_memory_stats()
    
    # Create the optimized config file
    config_path = create_optimized_config()
    
    # Print command to run the training manually
    print("\nTo run the optimized training, execute this command:")
    print(f"python main_TSP_edge_classification.py --config {config_path} --model GatedGCN --dataset TSP")
    print("\nRunning now...\n")
    
    # Run the command
    command = [
        "python", "main_TSP_edge_classification.py",
        "--config", config_path,
        "--model", "GatedGCN",
        "--dataset", "TSP"
    ]
    subprocess.run(command)
    
if __name__ == "__main__":
    main()