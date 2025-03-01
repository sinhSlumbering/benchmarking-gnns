# Training in Chunks

This approach helps when the dataset is too large to fit in memory:

1. Edit your config file to include:
   ```json
   "params": {
       ...
       "chunk_size": 5000,
       "epochs_per_chunk": 5
       ...
   }
   ```
2. Use the function train_val_pipeline_chunked:
   ```bash
   python main_TSP_edge_classification.py --model GCN --dataset TSP --config config.json
   ```
3. Confirm GPU memory usage is reduced. Each chunk loads a partial dataset, finishing training before moving to the next chunk.

