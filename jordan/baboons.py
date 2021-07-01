# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# read the data
df = pd.read_csv(r"C:\Users\Jordan\Documents\SPACT_REU_Research\multi_linear_research\jordan\Data\RFID_data.txt", delimiter="\t")


# vars
start_time = df['t'].iloc[0] - 20   # '-20' so the intervals work out
end_time = df['t'].iloc[-1]
time_interval = 60  # seconds (1 min)
options = {
    "node_size": 100,
    "arrowstyle": "-|>",
    "arrowsize": 12
}

# %%
# create the graph
G = nx.Graph()
G_over_time = []

# get baboon names
column_values = df[["i", "j"]].values.ravel()
unique_values = pd.unique(column_values)

# add nodes
num_nodes = len(unique_values)
G.add_nodes_from(range(num_nodes))

# name nodes
for i in range(len(G.nodes)):
    G.nodes[i]["name"] = unique_values[i]

# what does this do?
G.nodes.data()


while start_time <= end_time:
    # remove old edges
    G.remove_edges_from(G.edges())

    # find the interactions of interest
    interactions_df = df[df['t'].between(start_time, start_time + time_interval - 1)]

    # debugging and viewing purposes
    # print( start_time, interactions_df)

    # find the nodes of interest
    for i in range(len(interactions_df['t'])):
        counter = 0
        for n in G.nodes.data():
            if n[1]['name'] == interactions_df.iloc[i]['i']:
                node1 = counter
            if n[1]["name"] == interactions_df.iloc[i]['j']:
                node2 = counter
            counter += 1

        # add the edges
        G.add_edge(node1, node2)
        G.edges.data()

    """
    # prints the adj matrix and the graph representation
    print(nx.adj_matrix(G).todense())
    # display graph
    pos = nx.spring_layout(G)  # pos = nx.nx_agraph.graphviz_layout(G)
    nx.draw(G, pos, **options)
    plt.show()
    """

    # add to time graph
    G_over_time.append(nx.adj_matrix(G).todense())

    # increment start time
    start_time += time_interval



































