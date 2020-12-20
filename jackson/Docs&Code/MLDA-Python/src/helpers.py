import numpy as np
from math import prod

def tensor_to_vector(tensor):
    
    p = prod(tensor[0][0].shape)
    N = len(tensor)
    shape = (N, p)
    v = np.zeros(shape)
    for n in range(N):
        v[n, :] = vec(tensor[n][0])

    return v

def vec(matrix):
    return matrix.flatten('F')

def extract_train(data, N):   

    train = np.zeros((N, len(data[0])))
    for i in range(N):
        train[i] = data[i]
    return train