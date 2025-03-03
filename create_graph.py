import networkx as nx
import random

# Create random graph
G = nx.gnp_random_graph(100, 1, directed=True)

# Add some weights
for u, v in G.edges():
    G[u][v]['weight'] = random.random()

# Save the graph
nx.write_gpickle(G, "random_graph.gpickle")