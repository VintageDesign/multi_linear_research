import numpy as np
from math import prod
from operator import mul
from helpers import *

class Model_LDS:
    def __init__(self, I, J):
        self.mu0 = np.random.randn(J)
        self.Q0 = np.identity(J)
        self.Q = np.identity(J)
        self.R = np.identity(I)
        self.A = self.__init_multilinear_op(J, J)
        self.C = self.__init_multilinear_op(I, J)

    def __init_multilinear_op(self, I, J):
        rC = np.random.randn(I, I)
        while np.linalg.matrix_rank(rC) < I:
            rC = np.random.randn(I, I)
        U, S, V = np.linalg.svd(rC)
        C = U[:, 0:J]
        return C

    def print(self):
        print(f"mu0: {self.mu0}")
        print(f"Q0: {self.Q0}")
        print(f"Q: {self.Q}")
        print(f"R: {self.R}")
        print(f"A: {self.A}")
        print(f"C: {self.C}")

def num_params(I, J, Type):
    
    if type(I) != tuple or type(J) != tuple:
        print(f"I is a {type(I)}")
        print(f"J is a {type(J)}")
        raise TypeError("I and J must be a tuple")

    return count_covar_params(J, Type.Q0) + count_covar_params(J, Type.Q) + count_covar_params(I, Type.R) + sum(tuple(map(mul, J, J))) + sum(tuple(map(mul, I, J)))

def count_covar_params(I, Type):
    if Type == 'Isotropic':
        return 1
    elif Type == 'Diag':
        return prod(I)
    elif Type == 'Full':
        return prod(I)**2
    else:
        return 0

def learn_lds(X, type, J_lds):
    I = len(X[0])
    N = len(X)
    M = 1
    J = J_lds + 1
    maxiter = 20
    epsilon = 1e-5
    model = Model_LDS(I, J)
    
    i = 0
    while i < maxiter:
        print(f"At iteration {i}")
        
        # TODO Put time elasphed here

        forward(X, model)
        
        i += 1

def forward(X, model):

    M = len(X[0])
    N = len(X)
    H = len(model.A)
    Ih = np.identity(H)
    
    mu = np.zeros(N)
    V = np.zeros(N)
    P = np.zeros(N)

    mu[0] = model.mu0
    V[0] = model.Q0
    logli = 0

    KP = model.Q0
    for i in range(N):
        if i != 0:
            P[i-1] = model.A * V[i-1] * model.A