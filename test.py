import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt




G_practice = nx.DiGraph()

weighted_edges = [[0,1,0.1],
                  [0,2,0.2],
                  [1,3,0.3],
                  [1,4,0.4],
                  [2,4,0.5],
                  [2,0,0.6],
                  [3,0,0.7],
                  [3,1,0.8],
                  [4,3,0.9],
                  [4,2,1.0]
                 ]

G_practice.add_weighted_edges_from(weighted_edges)



print(np.prod([4,4]))