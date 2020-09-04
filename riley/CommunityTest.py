#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 09:42:10 2020

@author: rhoover
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms import community
from networkx.generators.community import LFR_benchmark_graph

##  Tensor packages for Tucker, etc.  ###
from sktensor.tucker import hooi
from sktensor import dtensor

### Standard Linear Algebra Package  ###
from scipy import linalg


plt.close()

N = 20
n = 100
p = 0.2

adj = np.zeros([n,n,2*N+2])

# Create random graph and store to the last lateral slice of adj
G = nx.fast_gnp_random_graph(n,p,seed = 4652, directed = False)
G_adj = nx.to_numpy_matrix(G)
adj[0:n,0:n,0] = G_adj # Stoer the first adjacency matrix in the first frontal slice of adj

# Create another random graph and store to the last lateral slice of adj
#G = nx.fast_gnp_random_graph(n,p,seed = 4652, directed = False)
#G_adj = nx.to_numpy_matrix(G)
adj[0:n,0:n,(2*N+2) - 1] = G_adj # Store the last adjacency matrix in the last frontal slice of adj

#  Generate graph community over 20 graphs.
for i in range(1,N+1):
    q = p-p*i/(N+1)
    P = np.array([[p,q],[q,p]])
    Gsbm = nx.to_numpy_matrix(nx.stochastic_block_model([int(n/2),int(n/2)],P))
    adj[0:n,0:n,i] = Gsbm
    print(i)

###  Generate back end (from communities to single community) - and plot in the loop
eps = 0.0
print('The j vector')
for j in range(1,N+2):
    q = p*(j-1)/(N+2) + eps#remove the eps for complete dissconnectivity #
    P = np.array([[p,q],[q,p]])
    Gsbm = nx.to_numpy_matrix(nx.stochastic_block_model([int(n/2),int(n/2)],P))
    adj[0:n,0:n,i+j] = Gsbm
    print(i+j)
   
    
####  Visualization of the graphs
plt.figure(1)
for i in range(2*N+2):
    plt.subplot(6,7,i+1)
    nx.draw(nx.from_numpy_matrix(adj[0:n,0:n,i]),node_size = 10)

####  Decompose using the Tucker decomposition  ####
T = dtensor(adj)
Y = hooi(T, [n, n, 2*N+2], init='nvecs')

U1 = Y[1][0]
U2 = Y[1][1]
U3 = Y[1][2]

plt.figure(2)
plt.subplot(1,2,1)
plt.imshow(U3.T)
plt.xlabel('Time')
plt.ylabel('Community Structure')
plt.title('Using HOSVD (Tucker Approximation)')

s_vec = np.zeros(2*N+2)
U_data = np.zeros([n,n,2*N+2])
V_data = np.zeros([n,n,2*N+2])
for i in range(2*N+2):
    U, s, Vh = np.linalg.svd(adj[0:n,0:n,i],full_matrices = True, compute_uv = True)
    s_vec[i] = s[0]
    U_data[0:n,0:n,i] = U
    V_data[0:n,0:n,i] = Vh


plt.subplot(1,2,2)
plt.plot(s_vec)
plt.xlabel('Time')
plt.ylabel('Largest Singular Value $\sigma_i$')
plt.title('Using Matrix SVD')

# Next largest, etc. #
for i in range(2*N+2):
    U, s, Vh = np.linalg.svd(adj[0:n,0:n,i],full_matrices = True, compute_uv = True)
    s_vec[i] = s[3]
    U_data[0:n,0:n,i] = U
    V_data[0:n,0:n,i] = Vh


plt.subplot(1,2,2)
plt.plot(s_vec)
plt.xlabel('Time')
plt.ylabel('Largest Singular Value $\sigma_i$')
plt.title('Using Matrix SVD')
#####################

plt.figure(3)
for i in range(N):
    plt.subplot(4,5,i+1)
    plt.scatter(U_data[0:n,0:1,i],U_data[0:n,1:2,i])

plt.suptitle('2-Dimensional Embedding From $U_{1,2}$')

plt.figure(4)
plt_cnt = 1
for i in range(N,2*N):
    plt.subplot(4,5,plt_cnt)
    plt.scatter(U_data[0:n,0:1,i],U_data[0:n,1:2,i])
    plt_cnt +=1

plt.suptitle('2-Dimensional Embedding From $U_{1,2}$')


