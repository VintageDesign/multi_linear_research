import numpy as np
import numpy.random as random

class LDS:
    def __init__(self, train) -> None:
        self.train = train
        self.shape = train.shape
        self.N = self.shape[0]
        self.I = self.shape[1]
        self.J = self.I
        self.M = 1
        self.maxiter = 20

    def fit(self):
        
        self.W = np.zeros(self.shape)
        self.mu0 = random.normal(size=self.J)
        self.Q0 = np.identity(self.J)
        self.Q = np.identity(self.J)
        self.R = np.identity(self.I)
        self.A = self.__init_multilinear_op(self.J)
        self.C = self.__init_multilinear_op(self.I)

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
            invSig = np.linalg.pinv(sigma_c)
            K = np.matmul(np.matmul(KP, np.transpose(self.C)), invSig)
            u_c = np.matmul(self.C, mu[i])
            delta = self.train[i] - u_c
            mu[i] += np.matmul(K, delta)
            V[i] = np.matmul((Ih - np.matmul(K, self.C)), KP)

        return mu, V, P


    def __backward(self, mu, V, P):
        
        Ez = np.zeros(mu.shape)
        Ezz = np.zeros(V.shape)
        Ezlz = np.zeros(V.shape)

        Ez[-1] = mu[-1]
        Vhat = V[-1]
        Ezz[-1] = Vhat + np.matmul(Ez[-1], np.transpose(Ez[-1]))

        for i in range(self.N-2, -1, -1):
            J = np.matmul(np.matmul(V[i], np.transpose(self.A)), np.linalg.inv(P[i]))
            Ez[i] = mu[i] + np.matmul(J, Ez[i+1] - np.matmul(self.A, mu[i]))
            Ezlz[i] = np.matmul(Vhat, np.transpose(J)) + np.matmul(Ez[i+1], np.transpose(Ez[i]))
            Vhat = V[i] + np.matmul(np.matmul(J, Vhat - P[i]), np.transpose(J))
            Ezz[i] = Vhat + np.matmul(Ez[i], np.transpose(Ez[i]))

        return Ez, Ezz, Ezlz

    def __MLE_mlds(self, Ez, Ezz, Ezlz):
        
        P = self.shape[1]
        H = Ez.shape[1]

        Szlz = np.sum(Ezlz, axis=0)
        Szz = np.sum(Ezz, axis=0)
        Sxz = np.zeros((P, H))
        for i in range(self.N):
            Sxz += np.matmul(self.train[i], np.transpose(Ez[i]))
        SzzN = Szz - Ezz[-1]

        self.mu0 = Ez[0]
        self.Q0 = Ezz[0] - np.matmul(Ez[0], np.transpose(Ez[0]))
        self.A = np.matmul(Szlz, np.linalg.inv(SzzN))
        tmp = np.matmul(self.A, np.transpose(Szlz))
        model.Q = (Szz - Ezz[0] - tmp - np.transpose(tmp) + np.matmul(np.matmul(self.A, SzzN), np.transpose(self.A))) / (self.N - 1)
        self.C = np.divide(Sxz, Szz)
        tmp = np.matmul(self.C, np.transpose(Sxz))
        self.R = np.matmul(np.transpose(self.train), self.train) - tmp - np.transpose(tmp) + np.matmul(np.matmul(self.C, Szz), np.transpose(self.C)) / self.N

    def __init_multilinear_op(self, I):
        rC = random.normal(size=(I, I))
        while np.linalg.matrix_rank(rC) < I:
            rC = random.normal(size=(I, I))
        u, s, v = np.linalg.svd(rC)
        return u

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

import time

import matplotlib.pyplot as plt
T = np.arange(2000, step=1)
print(T)
data = np.zeros((2000, 5))
data[0] = random.normal(size=5)
for i in range(1, 2000):
    for j in range(5):
        data[i][j] = np.sin(i) + random.normal()
print(data.shape)
plt.plot(data)
plt.xlim(0, 100)
plt.show()

train = data[:1800]
test = data[1800:]

model = LDS(train)
start = time.time()
model.fit()
end = time.time()
print(end-start)
forecast = model.forecast(200)

norm = np.abs((np.linalg.norm(forecast, axis=0) - np.linalg.norm(test, axis=0)) / np.linalg.norm(test, axis=0))

plt.figure()
plt.plot(test, color="blue")
plt.plot(forecast, color="red")
plt.show()

plt.plot(norm, color="red")
plt.show()