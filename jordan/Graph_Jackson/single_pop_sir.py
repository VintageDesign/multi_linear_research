# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import networkx as nx
import matplotlib.pyplot as plt
import numpy.random as random


# %%
color_map = {"s": "blue", "i": "red", "r": "gray"}
options = {
    "node_size": 100,
    "arrowstyle": "-|>",
    "arrowsize": 12
}


# %%
beta = 0.50
gamma = 0.50
N = 20
random.seed(1)


# %%
G = nx.Graph()
G.add_nodes_from(range(N))
for i in range(len(G.nodes)):
    G.nodes[i]["group"] = "s"
G.nodes[0]["group"] = "i"
G.nodes[0]["recovery_time_left"] = 1
G.nodes.data()


# %%
# Randomly make edges
for i in range(35):

    # Randomly decide two nodes
    node1 = random.randint(N)
    node2 = random.randint(N)
    while node2 == node1:
        node2 = random.randint(N)
    G.add_weighted_edges_from([(node1, node2, 0)])


# %%
for node in G.nodes.data():
    if node[1]["group"] == "i":
        # All other nodes are suspected to be infected!
        for other_node in G.neighbors(node[0]):
            G[node[0]][other_node]["weight"] = beta
G.edges.data()


# %%
G_over_time = []


# %%
def plot_G(G):
    pos=nx.spring_layout(G) # pos = nx.nx_agraph.graphviz_layout(G)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx(G, pos, node_color = [color_map[G.nodes[i]["group"]] for i in G], **options)
    #nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
plot_G(G)


# %%
def infect_neighbors(G):
    for node in G.nodes.data():
        if node[1]["group"] == "i" and node[1]["recovery_time_left"] != 1:
            
            # Gives a chance to be infected with all neighbors
            for neighbor in G.neighbors(node[0]):
                if G.nodes[neighbor]["group"] == "s" and random.rand() < beta:
                    # neighbor got infected :(
                    G[node[0]][neighbor]["weight"] = 0
                    G.nodes[neighbor]["group"] = "i"
                    G.nodes[neighbor]["recovery_time_left"] = 1

                    # Updates the neighbors
                    for infected_neighbor in G.neighbors(neighbor):
                        if G.nodes[infected_neighbor]["group"] == "s":
                            G[neighbor][infected_neighbor]["weight"] = beta

def recover_infected(G):
    for node in G.nodes.data():
        if node[1]["group"] == "i":
            # Add to the recovery rate
            node[1]["recovery_time_left"] -= gamma
            
            # Checks if they have recovered
            if node[1]["recovery_time_left"] < 0:
                # They are recovered
                node[1]["group"] = "r"

                # Updates all of the neighbors
                for neighbor in G.neighbors(node[0]):
                    G[node[0]][neighbor]["weight"] = 0

for i in range(10): 

    # Infect new individuals
    infect_neighbors(G)

    # See which infected get recovered
    recover_infected(G)

    G_over_time.append(G.copy())


# %%
fig, axs = plt.subplots(len(G_over_time), figsize = (5, 25))
pos = nx.spring_layout(G_over_time[0])
for i in range(len(G_over_time)):
    curr_G = G_over_time[i]
    nx.draw_networkx(curr_G, pos, node_color = [color_map[curr_G.nodes[i]["group"]] for i in curr_G], **options, ax = axs[i])


# %%
from IPython.display import HTML
fig, ax = plt.subplots()
def animate(i):
    ax.clear()
    curr_G = G_over_time[i]
    nx.draw_networkx(curr_G, pos, node_color = [color_map[curr_G.nodes[i]["group"]] for i in curr_G], **options, ax = ax)
ani = FuncAnimation(fig, animate, frames = len(G_over_time), interval = 1000)
HTML(ani.to_jshtml())


# %%
nx.adjacency_matrix(G_over_time[0]).todense()


