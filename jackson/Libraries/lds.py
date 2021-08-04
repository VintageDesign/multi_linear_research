import numpy as np
import numpy.random as random
import scipy as sp

class LDS:
    def __init__(self, train, J) -> None:
        self.train = train
        self.shape = train.shape
        self.N = self.shape[0]
        self.I = self.shape[1]
        self.J = J
        self.M = 1
        self.maxiter = 20

    def fit(self):
        
        self.W = np.zeros(self.shape)
        self.mu0 = random.normal(size=self.J)
        self.Q0 = np.identity(self.J)
        self.Q = np.identity(self.J)
        self.R = np.identity(self.I)
        self.A = self.__init_multilinear_op(self.J, self.J)
        self.C = self.__init_multilinear_op(self.I, self.J)

        for i in range(self.maxiter):
            mu, V, P = self.__forward()
            Ez, Ezz, Ezlz = self.__backward(mu, V, P)
            self.__MLE_mlds(Ez, Ezz, Ezlz)

    def __forward(self):
        H = len(self.A)
        Ih = np.identity(H)

        mu = np.zeros((self.N, len(self.mu0)))
        mu[0] = self.mu0
        V = np.zeros((self.N, self.Q0.shape[0], self.Q0.shape[1]))
        V[0] = self.Q0
        P = np.zeros((self.N, self.Q.shape[0], self.Q.shape[1]))
        
        for i in range(self.N):
            if i == 0:
                KP = self.Q0
            else:
                P[i-1] = np.matmul(np.matmul(self.A, V[i-1]), np.transpose(self.A)) + self.Q
                KP = P[i-1]
                mu[i] = np.matmul(self.A, mu[i-1])
            sigma_c = np.matmul(np.matmul(self.C, KP), np.transpose(self.C)) + self.R
            invSig = sp.linalg.pinv(sigma_c)
            K = np.matmul(np.matmul(KP, np.transpose(self.C)), invSig)
            u_c = np.matmul(self.C, mu[i])
            delta = self.train[i] - u_c
            mu[i] += np.matmul(K, delta)
            V[i] = np.matmul(Ih - np.matmul(K, self.C), KP)

        return mu, V, P


    def __backward(self, mu, V, P):
        
        Ez = np.zeros(mu.shape)
        Ezz = np.zeros(V.shape)
        Ezlz = np.zeros(V.shape)

        Ez[-1] = mu[-1]
        Vhat = V[-1]
        Ezz[-1] = Vhat + np.outer(Ez[-1], Ez[-1])

        for i in range(self.N-2, -1, -1):
            J = np.matmul(np.matmul(V[i], np.transpose(self.A)), np.linalg.inv(P[i]))
            Ez[i] = mu[i] + np.matmul(J, Ez[i+1] - np.matmul(self.A, mu[i]))
            Ezlz[i] = np.matmul(Vhat, np.transpose(J)) + np.outer(Ez[i+1], Ez[i])
            Vhat = V[i] + np.matmul(np.matmul(J, Vhat - P[i]), np.transpose(J))
            Ezz[i] = Vhat + np.outer(Ez[i], Ez[i])

        return Ez, Ezz, Ezlz

    def __MLE_mlds(self, Ez, Ezz, Ezlz):
        
        P = self.shape[1]
        H = Ez.shape[1]

        Szlz = np.zeros((H, H))
        Szz = np.zeros((H, H))
        Sxz = np.zeros((P, H))
        for i in range(self.N-1):
            Szlz += Ezlz[i]
        for i in range(self.N):
            Szz += Ezz[i]
            Sxz += np.outer(self.train[i], Ez[i])
        SzzN = Szz - Ezz[-1]

        self.mu0 = Ez[0]
        self.Q0 = Ezz[0] - np.outer(Ez[0], Ez[0])
        self.A = np.matmul(Szlz, np.linalg.inv(SzzN))
        tmp = np.matmul(self.A, np.transpose(Szlz))
        model.Q = (Szz - Ezz[0] - tmp - np.transpose(tmp) + np.matmul(np.matmul(self.A, SzzN), np.transpose(self.A))) / (self.N - 1)
        self.C = np.matmul(Sxz, np.linalg.inv(Szz))
        tmp = np.matmul(self.C, np.transpose(Sxz))
        self.R = (np.matmul(np.transpose(self.train), self.train) - tmp - np.transpose(tmp) + np.matmul(np.matmul(self.C, Szz), np.transpose(self.C))) / self.N

    def __init_multilinear_op(self, I, J):
        rC = random.normal(size=(I, I))
        while np.linalg.matrix_rank(rC) < I:
            rC = random.normal(size=(I, I))
        u, s, v = np.linalg.svd(rC)
        return u[:,:J]

    def forecast(self, interval):
        
        mu, V, P = self.__forward()
        Ez, Ezz, Ezlz = self.__backward(mu, V, P)
        self.mu0 = np.matmul(self.A, Ez[-1])
        
        H = Ez.shape[1]
        Ih = np.identity(H)

        forecast = np.zeros((interval, self.shape[1]))
        mu = np.zeros((interval, len(self.mu0)))
        mu[0] = self.mu0
        V = np.zeros((interval, self.Q0.shape[0], self.Q0.shape[1]))
        V[0] = self.Q0
        P = np.zeros((interval, self.Q.shape[0], self.Q.shape[1]))

        for i in range(interval):
            if i == 0:
                KP = self.Q0
            else:
                P[i-1] = np.matmul(np.matmul(self.A, V[i-1]), np.transpose(self.A)) + self.Q
                KP = P[i-1]
                mu[i] = np.matmul(self.A, mu[i-1])
            sigma_c = np.matmul(np.matmul(self.C, KP), np.transpose(self.C)) + self.R
            invSig = np.linalg.pinv(sigma_c)
            K = np.matmul(np.matmul(KP, np.transpose(self.C)), invSig)
            forecast[i] = np.matmul(self.C, mu[i])
            V[i] = np.matmul(Ih - np.matmul(K, self.C), KP)
        
        return forecast

    def single_step_forecast(self, interval, test):
        
        mu, V, P = self.__forward()
        Ez, Ezz, Ezlz = self.__backward(mu, V, P)
        self.mu0 = np.matmul(self.A, Ez[-1])
        
        H = Ez.shape[1]
        Ih = np.identity(H)

        forecast = np.zeros((interval, self.shape[1]))
        mu = np.zeros((interval, len(self.mu0)))
        mu[0] = self.mu0
        V = np.zeros((interval, self.Q0.shape[0], self.Q0.shape[1]))
        V[0] = self.Q0
        P = np.zeros((interval, self.Q.shape[0], self.Q.shape[1]))

        for i in range(interval):
            if i == 0:
                KP = self.Q0
            else:
                P[i-1] = np.matmul(np.matmul(self.A, V[i-1]), np.transpose(self.A)) + self.Q
                KP = P[i-1]
                mu[i] = np.matmul(self.A, mu[i-1])
            sigma_c = np.matmul(np.matmul(self.C, KP), np.transpose(self.C)) + self.R
            invSig = np.linalg.pinv(sigma_c)
            K = np.matmul(np.matmul(KP, np.transpose(self.C)), invSig)
            forecast[i] = np.matmul(self.C, mu[i])
            delta = test[i] - forecast[i]
            mu[i] += np.matmul(K, delta)
            V[i] = np.matmul((Ih - np.matmul(K, self.C)), KP)
        
        return forecast

    def print_model_parameters(self):
        print("mu0", self.mu0)
        print("Q0", self.Q0)
        print("Q", self.Q)
        print("R", self.R)
        print("R Shape", self.R.shape)
        print("A", self.A)
        print("A Shape", self.A.shape)
        print("C", self.C)
        print("C Shape", self.C.shape)
        input()

import time

import matplotlib.pyplot as plt

import scipy.io as sio
import sys

import pandas as pd
import numpy as np

import statsmodels.tsa.stattools as tsa
from statsmodels.tsa.api import VAR

sys.path.insert(0, '../../Libraries')
import JacksonsTSPackage as jts

sst = sio.loadmat(r'F:\repos\multi_linear_research\jackson\Docs&Code\L-TVAR\data\sst.mat')
tensor_shape = (len(sst['X']), sst['X'][0][0].shape[0], sst['X'][0][0].shape[1])
tensor_data = np.zeros(tensor_shape)
for i in range(tensor_shape[0]):
    tensor_data[i] = sst['X'][i][0]

N = len(tensor_data)
N_train = 1800
N_test = N - N_train
print(f"N: {N}")
print(f"N_train: {N_train}")
print(f"N_test: {N_test}")

X_vectorized = jts.tensor_to_vector(tensor_data)
train = jts.extract_train(X_vectorized, N_train)
test = jts.extract_test(X_vectorized, N_train, N_test)

print("Vector Shape:", X_vectorized.shape)

model = LDS(train, 3)
start = time.time()
model.fit()
end = time.time()
print(end - start)
forecast = model.single_step_forecast(200, test)

# print("FINAL MODEL")
# model.print_model_parameters()

def calc_norm(tensor):
    N = len(tensor)
    norm = np.zeros(N)
    for i in range(N):
        norm[i] = np.linalg.norm(tensor[i])
    return norm

plt.figure()
plt.plot(calc_norm(test), color="blue")
plt.plot(calc_norm(forecast), color="red")
plt.show()

error = np.abs((np.linalg.norm(forecast, axis=0) - np.linalg.norm(test, axis=0)) / np.linalg.norm(test, axis=0))

plt.plot(error, color="red")
plt.show()