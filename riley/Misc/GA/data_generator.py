import networkx as nx
import numpy as np
from networkx.linalg import *

def generate_initial_matrix(graph_size):
    """
    Generates the an n x n adj graph with a given level of community separation

    :param graph_size: the size of the graph.

    :return: Adj matrix with 2 seperate communities.
    """
    N = 20  # Time steps
    n = graph_size  # Nodes
    p = .4
    c = 2

    adj = np.zeros([n, n, 2 * N + 2])
    # Create random graph and store to the last lateral slice of adj
    G = nx.fast_gnp_random_graph(n, p, seed=4652, directed=False)
    G_adj = nx.to_numpy_matrix(G)
    adj[0:n, 0:n, 0] = G_adj  # Stoer the first adjacency matrix in the first frontal slice of adj
    # Create another random graph and store to the last lateral slice of adj
    # G = nx.fast_gnp_random_graph(n,p,seed = 4652, directed = False)
    # G_adj = nx.to_numpy_matrix(G)
    adj[0:n, 0:n, (2 * N + 2) - 1] = G_adj  # Store the last adjacency matrix in the last frontal slice of adj
    #  Generate graph community over 20 graphs.
    for i in range(1, N + 1):
        q = p - p * i / (N + 1)
        P = np.array([[p, q], [q, p]])
        Gsbm = nx.to_numpy_matrix(nx.stochastic_block_model([int(n / c), int(n / c)], P))
        adj[0:n, 0:n, i] = Gsbm
        # print(i)
    eps = 0.0
    for j in range(1, N + 2):
        q = p * (j - 1) / (N + 2) + eps  # remove the eps for complete dissconnectivity #
        P = np.array([[p, q], [q, p]])
        Gsbm = nx.to_numpy_matrix(nx.stochastic_block_model([int(n / c), int(n / c)], P))
        adj[0:n, 0:n, i + j] = Gsbm
        # print(i+j)

    return adj[:, :, 20]
