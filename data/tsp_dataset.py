import time
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors

import dgl
import torch
from torch.utils.data import Dataset
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import logging
import psutil
import os

class TSP(Dataset):
    def __init__(self, data_dir, split="train", num_neighbors=99, max_samples=100000, num_workers=None):    
        self.data_dir = data_dir
        self.split = split
        self.filename = f'{data_dir}/tsp100-100_{split}.txt'
        self.max_samples = max_samples
        self.num_neighbors = num_neighbors
        self.is_test = split.lower() in ['test', 'val']
        self.num_workers = num_workers if num_workers else cpu_count()
        
        self.graph_lists = []
        self.edge_labels = []
        self._prepare()
        self.n_samples = len(self.edge_labels)

    def _prepare(self):
        logging.info('Preparing graphs for the %s set using %d workers...' % (self.split.upper(), self.num_workers))
        
        start_time = time.time()

        # Read all lines from the file
        with open(self.filename, "r") as f:
            file_data = f.readlines()[:self.max_samples]
        
        num_lines = len(file_data)
        logging.info(f"Total graphs to process: {num_lines}")

        # Create temporary directory for storing processed graphs
        temp_dir = os.path.join(self.data_dir, 'temp', self.split)
        os.makedirs(temp_dir, exist_ok=True)

        # Use multiprocessing Pool to process data lines in parallel
        with Pool(self.num_workers) as pool:
            args_list = list(enumerate(file_data))
            for _ in tqdm(pool.imap_unordered(self._process_line, args_list), total=num_lines, desc=f"Processing {self.split}"):
                pass  # No need to collect results here

        # Collect the graphs after processing
        logging.info("Collecting processed graphs...")
        graph_files = sorted(os.listdir(temp_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))

        for graph_file in tqdm(graph_files, desc="Loading graphs"):
            with open(os.path.join(temp_dir, graph_file), 'rb') as f:
                g, labels = pickle.load(f)
                self.graph_lists.append(g)
                self.edge_labels.append(labels)

        # Optionally, delete the temporary files after loading
        import shutil
        shutil.rmtree(temp_dir)

        logging.info(f"Processed {len(self.graph_lists)} graphs in {time.time() - start_time:.2f} seconds.")

        # Log memory usage
        process = psutil.Process()
        mem_info = process.memory_info()
        logging.info(f"Memory Usage after processing {self.split} set: {mem_info.rss / (1024 ** 2):.2f} MB")

    def _process_line(self, args):
        index, line = args
        line = line.strip().split(" ")
        num_nodes = int(line.index('output') // 2)

        # Node coordinates
        nodes_coord = np.array(
            [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * num_nodes, 2)],
            dtype=np.float32
        )

        # Compute k-nearest neighbors using efficient methods
        nn = NearestNeighbors(n_neighbors=self.num_neighbors + 1, metric='euclidean')
        nn.fit(nodes_coord)
        distances, knns = nn.kneighbors(nodes_coord)

        # Remove self-loops if present
        knns = knns[:, 1:]  # Exclude the first neighbor (itself)
        distances = distances[:, 1:]

        # Tour nodes
        tour_nodes = np.array([int(node) - 1 for node in line[line.index('output') + 1:-1]], dtype=np.int32)

        # Edge adjacency matrix representation of tour
        edges_target = np.zeros((num_nodes, num_nodes), dtype=np.int32)
        edges_target[tour_nodes[:-1], tour_nodes[1:]] = 1
        edges_target[tour_nodes[1:], tour_nodes[:-1]] = 1
        edges_target[tour_nodes[-1], tour_nodes[0]] = 1
        edges_target[tour_nodes[0], tour_nodes[-1]] = 1

        # Global max and min weights
        global_max_weight = distances.max()
        global_min_weight = max(distances.min(), 1e-9)

        # Edge features and labels
        edge_feats = []
        edge_labels = []
        src_nodes = []
        dst_nodes = []

        for idx in range(num_nodes):
            neighbors = knns[idx]
            neighbor_distances = distances[idx]

            # Max and min weights among neighbors
            max_weight = max(neighbor_distances.max(), 1e-9)
            min_weight = max(neighbor_distances.min(), 1e-9)

            for n_idx, weight in zip(neighbors, neighbor_distances):
                if n_idx != idx:  # No self-connection
                    src_nodes.append(idx)
                    dst_nodes.append(n_idx)

                    # Edge features
                    edge_feats.append([
                        weight / global_max_weight,
                        weight / max_weight,
                        min_weight / max(weight, 1e-9),
                        global_min_weight / max(weight, 1e-9)
                    ])

                    edge_labels.append(edges_target[idx, n_idx])

        # Convert lists to tensors
        src_nodes = torch.tensor(src_nodes, dtype=torch.int64)
        dst_nodes = torch.tensor(dst_nodes, dtype=torch.int64)

        g = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)
        g.ndata['feat'] = torch.tensor(nodes_coord, dtype=torch.float32)

        # Add edge features and labels
        g.edata['feat'] = torch.tensor(edge_feats, dtype=torch.float32)
        g.edata['label'] = torch.tensor(edge_labels, dtype=torch.int64)

        # Save the graph and labels to a file
        temp_dir = os.path.join(self.data_dir, 'temp', self.split)
        output_file = os.path.join(temp_dir, f'graph_{index}.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump((g, edge_labels), f)

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Returns
            -------
            (dgl.DGLGraph, list)
                DGLGraph with node feature stored in `feat` field
                And a list of labels for each edge in the DGLGraph.
        """
        return self.graph_lists[idx], self.edge_labels[idx]
    



#################################
# class EdgeClassificationDataset(Dataset):
#     def __init__(self, graph_lists, edge_labels):
#         self.graph_lists = graph_lists
#         self.edge_labels = edge_labels
#         self.n_samples = len(self.graph_lists)

#     def __len__(self):
#         return self.n_samples

#     def __getitem__(self, idx):
#         return self.graph_lists[idx], self.edge_labels[idx]

#     @staticmethod
#     def collate(samples):
#         graphs, labels = map(list, zip(*samples))
#         # Edge classification labels need to be flattened to 1D tensor
#         labels = torch.LongTensor(np.concatenate(labels))
#         batched_graph = dgl.batch(graphs)
#         return batched_graph, labels

# class TSPDataset:
#     def __init__(self, data_dir, name="TSP"):
#         start = time.time()
#         print("[I] Loading dataset %s..." % (name))
#         self.name = name
#         self.data_dir = data_dir  # e.g., 'data/TSP/'
#         self.train = self._load_split('train')
#         self.val = self._load_split('val')
#         self.test = self._load_split('test')
#         print('train, val, test sizes:', len(self.train), len(self.val), len(self.test))
#         print("[I] Finished loading.")
#         print("[I] Data load time: {:.4f}s".format(time.time() - start))
        
#     def _load_split(self, split):
#         split_pkl_file = os.path.join(self.data_dir, f'{self.name}_{split}.pkl')
#         if not os.path.exists(split_pkl_file):
#             raise FileNotFoundError(f"Pickle file {split_pkl_file} not found.")
#         with open(split_pkl_file, 'rb') as f:
#             graph_lists, edge_labels = pickle.load(f)
#         # Create a Dataset object from the data
#         dataset = EdgeClassificationDataset(graph_lists, edge_labels)
#         return dataset

#     @staticmethod
#     def collate(samples):
#         # The input samples is a list of pairs (graph, label).
#         graphs, labels = map(list, zip(*samples))
#         # Edge classification labels need to be flattened to 1D tensor
#         labels = torch.LongTensor(np.concatenate(labels))
#         batched_graph = dgl.batch(graphs)
#         return batched_graph, labels

#     # If you need dense tensors for specific GNN models, you can implement this method as needed.
#     # def collate_dense_gnn(self, samples, edge_feat):
#     #     pass  # Implement as needed
class EdgeClassificationDataset(Dataset):
    def __init__(self, graph_lists, edge_labels):
        self.graph_lists = graph_lists
        self.edge_labels = edge_labels
        self.n_samples = len(self.graph_lists)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.graph_lists[idx], self.edge_labels[idx]

    @staticmethod
    def collate(samples):
        graphs, labels = map(list, zip(*samples))
        # Edge classification labels need to be flattened to 1D tensor
        labels = torch.LongTensor(np.concatenate(labels))
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels

class TSPDataset:
    def __init__(self, data_dir, name="TSP"):
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        self.data_dir = 'data/TSP/'  # e.g., 'data/TSP/'
        combined_pkl_file = os.path.join(self.data_dir, f'{self.name}.pkl')
        if not os.path.exists(combined_pkl_file):
            raise FileNotFoundError(f"Pickle file {combined_pkl_file} not found.")
        with open(combined_pkl_file, 'rb') as f:
            print(f)
            datasets = pickle.load(f)
            # datasets is a list of tuples: [(train_graphs, train_labels), (val_graphs, val_labels), (test_graphs, test_labels)]
            # So we can unpack them accordingly
            if len(datasets) == 3:
                train_data = datasets[0]
                val_data = datasets[1]
                test_data = datasets[2]
            else:
                raise ValueError("Unexpected data format in the pickle file.")
        # Create Dataset objects for each split
        self.train = EdgeClassificationDataset(*train_data)
        self.val = EdgeClassificationDataset(*val_data)
        self.test = EdgeClassificationDataset(*test_data)
        print('Train, Val, Test sizes:', len(self.train), len(self.val), len(self.test))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

    @staticmethod
    def collate(samples):
        graphs, _ = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        labels = batched_graph.edata['label']
        return batched_graph, labels

    # If you need dense tensors for specific GNN models, you can implement this method as needed.
    # def collate_dense_gnn(self, samples, edge_feat):
    #     pass  # Implement as needed
