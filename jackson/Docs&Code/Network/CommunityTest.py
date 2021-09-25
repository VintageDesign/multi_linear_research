#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 09:42:10 2020

@author: rhoover
"""


from matplotlib import animation
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from networkx.algorithms.shortest_paths.unweighted import predecessor
import numpy as np
import pandas as pd
from networkx.algorithms import community
from networkx.generators.community import LFR_benchmark_graph



### Standard Linear Algebra Package  ###
import numpy.linalg as la


N = 1000
n = 90
p = 0.2
period = 200


###  Create the graphs
G = nx.fast_gnp_random_graph(n,p,seed = 4652, directed = False)
G_adj = nx.to_numpy_matrix(G)
tensor = np.zeros((N, n, n))
qs = np.zeros(N)
rs = np.zeros(N)
for i in range(N):
    if i % period < period // 4 or i % period > 3 * period // 4:
        q = p*(np.cos(2*i*(2*np.pi/period))+1)/2
        r = p
    else:
        q = 0
        r = p*(np.cos(2*(i-period//4)*(2*np.pi/period))+1)/2
    P = np.array(
        [[p,q,q],
        [q,p,r],
        [q,r,p]])
    Gsbm = nx.to_numpy_matrix(nx.stochastic_block_model([n//3,n//3,n//3],P))
    tensor[i] = Gsbm
    qs[i] = q
    rs[i] = r

plt.plot(range(N), np.repeat(p, N), "--")
plt.plot(range(N), qs)
plt.plot(range(N), rs)
plt.legend(["p", "q", "r"], loc="upper right")
plt.show()

#color_map = []
# for node in G:
#     if node <= n//2:
#         color_map.append('blue')
#     else:
#         color_map.append('orange')

###  Visualization of the graphs
###  Plot the norm of the graph
norm = np.array([la.norm(adj) for adj in tensor])
plt.plot(np.arange(len(tensor)), norm)
plt.show()

###  Animate the graphs
fig, ax = plt.subplots()
def animate(i):
    ax.clear()
    nx.draw_circular(nx.from_numpy_matrix(tensor[i]), node_size=10) #, with_labels=True, node_color=color_map
ani = FuncAnimation(fig, animate, frames = len(tensor), interval=100)
#ani.save("graph.gif")
plt.show()

###  Animate the adj matrix of tensor
def animate_tensor(tensor, N, save_path):
    global im
    global display_tensor
    display_tensor = tensor
    fig = plt.figure()
    im = plt.imshow(display_tensor[0], cmap=plt.get_cmap("gray"), animated = True)
    ani = FuncAnimation(fig, animate, interval = 100, frames = N, blit=True)
    plt.show()
    #ani.save(save_path)

def animate(i):
    global im
    im.set_array(display_tensor[i])
    return im,

animate_tensor(tensor, 200, "adj_mat.gif")


###  Create model
# N_train = 800
# N_test = 200
# train_tensor = jts.extract_train_tensor(tensor, N_train)
# test_tensor = jts.extract_test_tensor(tensor, N_train, N_test)

# interval = 100
# ltar = LTAR(diff(train_tensor, 100))
# ltar.fit(20, "dct")
# result_tensor = invert_diff(ltar.forecast(N_test), train_tensor, interval)


# ###  Plot error 
# error = jts.calc_mape_per_matrix(test_tensor, result_tensor)
# error.plot()
# plt.show()

# ###  Plot the norm of the forecasted graph
# forecasted_norm = np.array([la.norm(adj) for adj in result_tensor])
# plt.plot(np.arange(len(result_tensor)), forecasted_norm)
# plt.show()

# ###  Plot the norms of the original and forecasted tensors
# def calc_norm(tensor):
#     norms = []
#     for i in range(len(tensor)):
#         norms.append(np.linalg.norm(tensor[i], ord="fro"))
#     return pd.DataFrame(norms)
# df = pd.concat([calc_norm(test_tensor), calc_norm(result_tensor)], axis=1)
# df.plot()
# plt.show()

# ###  Adj matrix for forecasted tensor
# animate_tensor(result_tensor, len(result_tensor), "forecast.gif")

# print(len(result_tensor))


###  Correcting the data values
# for i in range(len(result_tensor)):
#     for j in range(n):
#         for k in range(n):
#             if result_tensor[i][j][k] <= 0.5:
#                 result_tensor[i][j][k] = 0
#             else:
#                 result_tensor[i][j][k] = 1


# ###  Visualization of the forecasted graphs
# fig, ax = plt.subplots()
# def animate_forecast(i):
#     ax.clear()
#     nx.draw_networkx(nx.from_numpy_matrix(result_tensor[i]), node_size=10, with_labels=False, node_color=color_map)
# ani = FuncAnimation(fig, animate_forecast, frames = len(result_tensor), interval=100)
# ani.save("forecasted_graph.gif")
# plt.show()
