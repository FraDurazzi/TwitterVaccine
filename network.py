import networkx as nx
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re,os
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import re, os
from sklearn.metrics import  accuracy_score, precision_score, recall_score, confusion_matrix
from DIRS import TRANSFORMERS_CACHE_DIR, DATA_DIR, LARGE_DATA_DIR, NETWORK_DATA
from build_graphs import compute_graph, main, parse_date
from fa2 import ForceAtlas2
import igraph


forceatlas2 = ForceAtlas2(
                        # Behavior alternatives
                        outboundAttractionDistribution=True,  # Dissuade hubs
                        linLogMode=False,  # NOT IMPLEMENTED
                        adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                        edgeWeightInfluence=1.0,

                        # Performance
                        jitterTolerance=1.0,  # Tolerance
                        barnesHutOptimize=True,
                        barnesHutTheta=1.2,
                        multiThreaded=False,  # NOT IMPLEMENTED

                        # Tuning
                        scalingRatio=2.0,
                        strongGravityMode=False,
                        gravity=1.0,

                        # Log
                        verbose=True)
G=nx.read_graphml(NETWORK_DATA+"retweet_graph_undirected_2024-01-01.graphml")
positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=200)
G1 = igraph.Graph.TupleList(G.edges(), directed=False)
layout = forceatlas2.forceatlas2_igraph_layout(G1, pos=None, iterations=200)
nx.draw_networkx_nodes(G, positions, node_size=20, node_color="blue", alpha=0.4)
nx.draw_networkx_edges(G, positions, edge_color="green", alpha=0.05)
plt.axis('off')
plt.show()
igraph.plot(G1,Layout=layout).show()
