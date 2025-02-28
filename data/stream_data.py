import os
import torch
import numpy as np
import dgl
import pickle
from torch.utils.data import Dataset, DataLoader

class StreamingGraphDataset(Dataset):
    """A dataset class that loads graphs from disk on-demand instead of holding all in memory."""
    
    def __init__(self, data_dir, split='train', transform=None, pre_transform=None, in_memory=False):
        """
        Args:
            data_dir (str): Directory containing the dataset files
            split (str): 'train', 'val', or 'test'
            transform (callable, optional): Transform to be applied on each graph
            pre_transform (callable, optional): Transform to be applied on all graphs before loading
            in_memory (bool): Whether to load all data in memory (False for streaming)
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.pre_transform = pre_transform
        self.in_memory = in_memory
        
        # Load metadata file
        meta_path = os.path.join(data_dir, f"{split}_metadata.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            # Create metadata by scanning the directory
            self._create_metadata()
        
        # If in_memory is True, load all graphs
        self.data = None
        if in_memory:
            self._load_all_data()
    
    def _create_metadata(self):
        """Scan the data directory to create metadata about the dataset."""
        self.metadata = {
            'num_samples': 0,
            'filenames': [],
            'class_distribution': {},
            'feature_dim': None,
        }
        
        data_files = sorted([f for f in os.listdir(os.path.join(self.data_dir, self.split))
                           if f.endswith('.pkl')])
        
        self.metadata['num_samples'] = len(data_files)
        self.metadata['filenames'] = data_files
        
        # Get feature dimensions from the first graph
        if data_files:
            sample_path = os.path.join(self.data_dir, self.split, data_files[0])
            with open(sample_path, 'rb') as f:
                sample = pickle.load(f)
                if isinstance(sample, dgl.DGLGraph):
                    self.metadata['feature_dim'] = sample.ndata['feat'].shape[1] if 'feat' in sample.ndata else 0
        
        # Save metadata
        with open(os.path.join(self.data_dir, f"{self.split}_metadata.pkl"), 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def _load_all_data(self):
        """Load all graphs into memory."""
        self.data = []
        for filename in self.metadata['filenames']:
            file_path = os.path.join(self.data_dir, self.split, filename)
            with open(file_path, 'rb') as f:
                graph = pickle.load(f)
                if self.pre_transform:
                    graph = self.pre_transform(graph)
                self.data.append(graph)
    
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.metadata['num_samples']
    
    def __getitem__(self, idx):
        """Get a graph by index."""
        if self.in_memory and self.data is not None:
            graph = self.data[idx]
        else:
            # Load from disk
            file_path = os.path.join(self.data_dir, self.split, self.metadata['filenames'][idx])
            with open(file_path, 'rb') as f:
                graph = pickle.load(f)
                if self.pre_transform:
                    graph = self.pre_transform(graph)
        
        if self.transform:
            graph = self.transform(graph)
            
        return graph
    
    def collate(self, samples):
        """Collate function to be used with DataLoader."""
        # For TSP edge classification, ensure we handle the expected format
        graphs = [item[0] if isinstance(item, tuple) else item for item in samples]
        labels = [item[1] if isinstance(item, tuple) and len(item) > 1 else None for item in samples]
        
        batched_graph = dgl.batch(graphs)
        
        if labels[0] is not None:
            labels = torch.stack(labels) if isinstance(labels[0], torch.Tensor) else torch.tensor(labels)
            return batched_graph, labels
        return batched_graph


class StreamingTSPDataset:
    """Wrapper class for TSP dataset with train/val/test splits."""
    
    def __init__(self, data_dir, in_memory={'train': False, 'val': False, 'test': False}):
        """
        Args:
            data_dir (str): Directory containing the dataset files
            in_memory (dict): Whether to load specific splits in memory
        """
        self.data_dir = data_dir
        self.name = os.path.basename(data_dir)
        
        # Create datasets for each split
        self.train = StreamingGraphDataset(data_dir, 'train', in_memory=in_memory.get('train', False))
        self.val = StreamingGraphDataset(data_dir, 'val', in_memory=in_memory.get('val', False))
        self.test = StreamingGraphDataset(data_dir, 'test', in_memory=in_memory.get('test', False))
        
        # Use the collate function from the train dataset
        self.collate = self.train.collate
