
# This is a quick fix that will dynamically check and adjust the function calls to match 
# the actual return values. Run this script to apply the fix.

import inspect
import re
import os

def fix_train_val_pipeline():
    # Path to the main file
    filepath = '/home/cailtsptest/tsp/benchmarking-gnns/main_TSP_edge_classification.py'
    
    # First, let's import the actual functions to see their signatures
    try:
        from train.train_TSP_edge_classification import train_epoch_sparse as train_epoch
        from train.train_TSP_edge_classification import evaluate_network_sparse as evaluate_network
        
        # Get the number of return values
        train_source = inspect.getsource(train_epoch)
        eval_source = inspect.getsource(evaluate_network)
        
        # Check how many values are returned
        train_returns = len(re.findall(r'return\s+([^#\n]+)', train_source)[0].split(','))
        eval_returns = len(re.findall(r'return\s+([^#\n]+)', eval_source)[0].split(','))
        
        print(f"train_epoch returns {train_returns} values")
        print(f"evaluate_network returns {eval_returns} values")
        
        # Read the file
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Fix the function calls
        if train_returns == 5:  # Based on the error message
            # Replace the unpacking with the correct number of return values
            content = re.sub(
                r'epoch_train_loss, epoch_train_f1, optimizer, train_correct_counts, train_total_counts, train_false_positives = train_epoch',
                r'epoch_train_loss, epoch_train_f1, optimizer, train_correct_counts, train_total_counts = train_epoch',
                content
            )
            
            # Add default values for the missing variable
            content = re.sub(
                r'train_correct_counts, train_total_counts = train_epoch',
                r'train_correct_counts, train_total_counts = train_epoch\n                train_false_positives = {}',
                content
            )
        
        # Fix evaluate_network calls similarly if needed
        if eval_returns == 5:
            content = re.sub(
                r'epoch_val_loss, epoch_val_f1, val_correct_counts, val_total_counts, val_false_positives = evaluate_network',
                r'epoch_val_loss, epoch_val_f1, val_correct_counts, val_total_counts = evaluate_network\n                val_false_positives = {}',
                content
            )
            content = re.sub(
                r'epoch_test_loss, epoch_test_f1, test_correct_counts, test_total_counts, test_false_positives = evaluate_network',
                r'epoch_test_loss, epoch_test_f1, test_correct_counts, test_total_counts = evaluate_network\n                test_false_positives = {}',
                content
            )
            content = re.sub(
                r'_, test_f1, test_correct_counts, test_total_counts, test_false_positives = evaluate_network',
                r'_, test_f1, test_correct_counts, test_total_counts = evaluate_network\n    test_false_positives = {}',
                content
            )
            content = re.sub(
                r'_, train_f1, train_correct_counts, train_total_counts, train_false_positives = evaluate_network',
                r'_, train_f1, train_correct_counts, train_total_counts = evaluate_network\n    train_false_positives = {}',
                content
            )
        
        # Write the fixed content
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"Fixed {filepath}")
    
    except ImportError:
        print("Could not import train functions to check signatures")
    except Exception as e:
        print(f"Error fixing the file: {e}")

if __name__ == "__main__":
    fix_train_val_pipeline()
