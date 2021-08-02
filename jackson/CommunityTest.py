#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 09:42:10 2020

@author: rhoover
"""

from Libraries.JacksonsTSPackage import forecast
from matplotlib import animation
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
from networkx.algorithms import community
from networkx.generators.community import LFR_benchmark_graph
import sys
sys.path.insert(0, 'Libraries')
import JacksonsTSPackage as jts
from ltar import LTAR, diff, invert_diff

### Standard Linear Algebra Package  ###
import numpy.linalg as la


N = 1000
n = 50
p = 0.2
period = 100

G = nx.fast_gnp_random_graph(n,p,seed = 4652, directed = False)
G_adj = nx.to_numpy_matrix(G)
tensor = np.zeros((N, n, n))
for i in range(N):
    q = p*(np.cos(i*(2*np.pi/period))+1)/2
    P = np.array([[p,q],[q,p]])
    Gsbm = nx.to_numpy_matrix(nx.stochastic_block_model([n//2,n//2],P))
    tensor[i] = Gsbm

####  Visualization of the graphs
# norm = np.array([la.norm(adj) for adj in tensor])
# plt.plot(np.arange(len(tensor)), norm)
# plt.show()

# fig, ax = plt.subplots()
# def animate(i):
#     ax.clear()
#     nx.draw_networkx(nx.from_numpy_matrix(tensor[i]), node_size=10, with_labels=False)
# ani = FuncAnimation(fig, animate, frames = len(tensor), interval=100)
# #ani.save("graph.gif")
# plt.show()

def animate_tensor(tensor, N, save_path):
    global im
    global display_tensor
    display_tensor = tensor
    fig = plt.figure()
    im = plt.imshow(display_tensor[0], cmap=plt.get_cmap("gray"), animated = True)
    ani = FuncAnimation(fig, animate, interval = 100, frames = N, blit=True)
    plt.show()
    ani.save(save_path)
def animate(i):
    global im
    im.set_array(display_tensor[i])
    return im,

#animate_tensor(tensor, 200, "adj_mat.gif")

N_train = 800
N_test = 200
train_tensor = jts.extract_train_tensor(tensor, N_train)
test_tensor = jts.extract_test_tensor(tensor, N_train, N_test)

interval = 100
ltar = LTAR(diff(train_tensor, 100))
ltar.fit(20, "dct")
result_tensor = invert_diff(ltar.forecast(N_test), train_tensor, interval)

error = jts.calc_mape_per_matrix(test_tensor, result_tensor)
error.plot()
plt.show()

def calc_norm(tensor):
    norms = []
    for i in range(len(tensor)):
        norms.append(np.linalg.norm(tensor[i], ord="fro"))
    return pd.DataFrame(norms)
df = pd.concat([calc_norm(test_tensor), calc_norm(result_tensor)], axis=1)
df.plot()
plt.show()

animate_tensor(result_tensor, len(result_tensor), "forecast.gif")