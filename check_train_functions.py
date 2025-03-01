
def check_train_function_signature():
    """
    This function inspects the actual signatures of the train and evaluate functions
    to help us understand what they return.
    """
    import inspect
    from train.train_TSP_edge_classification import train_epoch_sparse, evaluate_network_sparse
    
    print("Inspecting train_epoch_sparse function:")
    print(inspect.signature(train_epoch_sparse))
    
    # Get source code if possible
    try:
        print("\nSource code:")
        print(inspect.getsource(train_epoch_sparse))
    except Exception as e:
        print(f"Could not get source code: {e}")
    
    print("\nInspecting evaluate_network_sparse function:")
    print(inspect.signature(evaluate_network_sparse)) 
    
    try:
        print("\nSource code:")
        print(inspect.getsource(evaluate_network_sparse))
    except Exception as e:
        print(f"Could not get source code: {e}")

if __name__ == "__main__":
    check_train_function_signature()
