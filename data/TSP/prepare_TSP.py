#!/usr/bin/env python
# coding: utf-8

# # Notebook for preparing and saving TSP graphs

# In[1]:


import numpy as np
import torch
import pickle
import time
import os

import matplotlib.pyplot as plt


# # Download TSP dataset

# In[2]:



#     get_ipython().system('curl https://www.dropbox.com/s/1wf6zn5nq7qjg0e/TSP.zip?dl=1 -o TSP.zip -J -L -k')
#     get_ipython().system('unzip TSP.zip -d ../')
#     # !tar -xvf TSP.zip -C ../
# else:
#     print('File already downloaded')
#


# # Convert to DGL format and save with pickle

# In[2]:


# import os
# os.chdir('../../') # go to root folder of the project
# print(os.getcwd())
#

# In[3]:


import pickle




from torch.utils.data import DataLoader

from data.data import LoadData
from data.TSP import TSP, TSPDatasetDGL, TSPDataset


# In[4]:


start = time.time()

DATASET_NAME = 'TSP'
dataset = TSPDatasetDGL(DATASET_NAME) 

print('Time (sec):',time.time() - start) # ~ 30 mins


# In[5]:



#     graph_sizes = []
#     for graph in dataset:
#         graph_sizes.append(graph[0].number_of_nodes())
#         #graph_sizes.append(graph[0].number_of_edges())
#     plt.figure(1)
#     plt.hist(graph_sizes, bins=20)
#     plt.title(title)
#     plt.show()
#     graph_sizes = torch.Tensor(graph_sizes)
#     print('nb/min/max :',len(graph_sizes),graph_sizes.min().long().item(),graph_sizes.max().long().item())
#
# plot_histo_graphs(dataset.train,'trainset')
# plot_histo_graphs(dataset.val,'valset')
# plot_histo_graphs(dataset.test,'testset')
#

# In[12]:


print(len(dataset.train))
print(len(dataset.val))
print(len(dataset.test))

print(dataset.train[0])
print(dataset.val[0])
print(dataset.test[0])


# In[6]:


start = time.time()

with open('data/TSP/TSP.pkl','wb') as f:
    pickle.dump([dataset.train,dataset.val,dataset.test],f)
        
print('Time (sec):',time.time() - start) # 58s


# # Test load function

# In[14]:


DATASET_NAME = 'TSP'
dataset = LoadData(DATASET_NAME)  # 20s
trainset, valset, testset = dataset.train, dataset.val, dataset.test


# In[15]:


start = time.time()

batch_size = 10
collate = TSPDataset.collate
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate)

print('Time (sec):',time.time() - start)  # 0.0003s


# # Plot TSP samples

# In[16]:

#
# from scipy.spatial.distance import pdist, squareform
# import matplotlib
# import matplotlib.pyplot as plt
# import networkx as nx
#
#
# def _edges_to_node_pairs(W):
#     """Helper function to convert edge matrix into pairs of adjacent nodes.
#     """
#     pairs = []
#     for r in range(len(W)):
#         for c in range(len(W)):
#             if W[r][c] == 1:
#                 pairs.append((r, c))
#     return pairs
#
# def plot_tsp(x_coord, W, tour):
#     """
#     Helper function to plot TSP tours.
#
#     Args:
#         x_coord: Coordinates of nodes
#         W: Graph as adjacency matrix
#         tour: Predicted tour
#         title: Title of figure/subplot
#     """
#     W_val = squareform(pdist(x_coord, metric='euclidean'))
#     G = nx.from_numpy_matrix(W_val)
#
#     pos = dict(zip(range(len(x_coord)), x_coord))
#
#     adj_pairs = _edges_to_node_pairs(W)
#
#     tour_pairs = []
#     for idx in range(len(tour)-1):
#         tour_pairs.append((tour[idx], tour[idx+1]))
#     tour_pairs.append((tour[idx+1], tour[0]))
#
#     node_size = 50/(len(x_coord)//50)
#
#     nx.draw_networkx_nodes(G, pos, node_color='b', node_size=node_size)
#     nx.draw_networkx_edges(G, pos, edgelist=adj_pairs, alpha=0.25, width=0.1)
#     nx.draw_networkx_edges(G, pos, edgelist=tour_pairs, alpha=1, width=1.5, edge_color='r')
#
#
# # In[17]:
#
#
# filename = "data/TSP/tsp50-500_test.txt"
# file_data = open(filename, "r").readlines()
# num_neighbors = 25
#
#
# # In[17]:
#
#
# for graph_idx, line in enumerate(file_data):
#     line = line.split(" ")  # Split into list
#     num_nodes = int(line.index('output')//2)
#
#     # Convert node coordinates to required format
#     nodes_coord = []
#     for idx in range(0, 2 * num_nodes, 2):
#         nodes_coord.append([float(line[idx]), float(line[idx + 1])])
#
#     # Compute distance matrix
#     W_val = squareform(pdist(nodes_coord, metric='euclidean'))
#     # Determine k-nearest neighbors for each node
#     knns = np.argpartition(W_val, kth=num_neighbors, axis=-1)[:, num_neighbors::-1]
#
#     W = np.zeros((num_nodes, num_nodes))
#     # Make connections 
#     for idx in range(num_nodes):
#         W[idx][knns[idx]] = 1
#
#     # Convert tour nodes to required format
#     # Don't add final connection for tour/cycle
#     tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]][:-1]
#
#     # Compute an edge adjacency matrix representation of tour
#     edges_target = np.zeros((num_nodes, num_nodes))
#     for idx in range(len(tour_nodes) - 1):
#         i = tour_nodes[idx]
#         j = tour_nodes[idx + 1]
#         edges_target[i][j] = 1
#         edges_target[j][i] = 1
#     # Add final connection of tour in edge target
#     edges_target[j][tour_nodes[0]] = 1
#     edges_target[tour_nodes[0]][j] = 1
#
#     if num_nodes == 498:
#         print(num_nodes)
#         print(tour_nodes)
#
#         plt.figure(figsize=(5,5))
#         plot_tsp(nodes_coord, W, tour_nodes)
#         plt.savefig(f"img/tsp{num_nodes}_{graph_idx}.pdf", format='pdf', dpi=1200, bbox_inches='tight')
#         plt.show()
#         print("Stop (y/n)")
#         if input() == 'y':
#             break
#
#
# # In[ ]:
#
#
#

