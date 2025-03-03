"""
Utility script to create a model_info file for an existing trained model.
This is useful when you have a model checkpoint but no corresponding info file.
"""

import pickle
import argparse
import os
import torch

def create_model_info(model_path, output_path=None):
    """
    Create a basic model_info dictionary and save it to a pickle file.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model file
    output_path : str, optional
        Path to save the model info file (default: model_path with '.info.pkl' extension)
    """
    if output_path is None:
        # Generate default output path
        dirname, filename = os.path.split(model_path)
        basename = os.path.splitext(filename)[0]
        output_path = os.path.join(dirname, f"{basename}_info.pkl")
    
    # Create basic model info
    model_info = {
        'net_params': {
            'L': 4,                # Number of layers
            'hidden_dim': 70,      # Hidden dimension
            'out_dim': 70,         # Output dimension
            'in_dim': 2,           # Input node feature dimension
            'in_dim_edge': 4,      # Input edge feature dimension - updated to 4 based on error
            'batch_norm': True,    # Use batch normalization
            'residual': True,      # Use residual connections
            'edge_feat': True,     # Use edge features
            'n_classes': 1,        # Binary classification
            'dropout': 0.0,        # Dropout rate
            'readout': "mean",     # Readout function
            'in_feat_dropout': 0.0 # Input feature dropout
        },
        'best_val_metric': 0.0,    # Placeholder for validation metric
        'test_metric': 0.0         # Placeholder for test metric
    }
    
    # Save the model info
    with open(output_path, 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"Created model info file at {output_path}")
    return model_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create model info file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save model info')
    args = parser.parse_args()
    
    create_model_info(args.model_path, args.output_path)
