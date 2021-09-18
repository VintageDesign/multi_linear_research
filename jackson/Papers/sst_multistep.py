import scipy.io as sio
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.tsa.stattools as tsa
from statsmodels.tsa.api import VAR
import statistics

sys.path.insert(0, '../Libraries')
import JacksonsTSPackage as jts
from ltar import LTAR
from mlds import LMLDS
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

import time

sst = sio.loadmat('F:\\repos\multi_linear_research\jackson\Docs&Code\L-TVAR\data\sst.mat')

N = len(sst['X'])
N_train = 1800
N_test = N - N_train
print(f"N: {N}")
print(f"N_train: {N_train}")
print(f"N_test: {N_test}")

tensor_shape = (len(sst['X']), sst['X'][0][0].shape[0], sst['X'][0][0].shape[1])

tensor_data = np.zeros(tensor_shape)
for i in range(tensor_shape[0]):
    tensor_data[i] = sst['X'][i][0]

def plot_norm(tensor, N):
    norms = []
    for i in range(N):
        norms.append(np.linalg.norm(tensor[i], ord="fro"))
    pd.DataFrame(norms).plot()
    plt.show()
plot_norm(tensor_data, N)

train_tensor = jts.extract_train_tensor(tensor_data, N_train)
test_tensor = jts.extract_test_tensor(tensor_data, N_train, N_test)

n_features = tensor_shape[1] * tensor_shape[2]
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        start_ix = i - n_steps
        # check if we are beyond the sequence
        if start_ix >= 0:
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[start_ix:i], sequence[i]
            X.append(seq_x)
            y.append(seq_y)
    return np.asarray(X), np.asarray(y)
n_steps = 5
sequence = tensor_data.reshape((N, tensor_shape[1] * tensor_shape[2]))
X, y = split_sequence(sequence, n_steps)
train_X = X[:N_train-5]
train_y = y[:N_train-5]
test_X = X[N_train-5:N_train+N_test-5]
test_y = y[N_train-5:N_train+N_test-5]
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
model = Sequential()
model.add(LSTM(100, activation='relu',return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mse')

lstm = []
ltar = []
lmlds = []
ltari = []
lstar = []
for i in range(20):
    print(i)
    start = time.time()
    model.fit(train_X, train_y, epochs=200, verbose=1)
    end = time.time()
    lstm.append(end-start)

    dct_ltar = LTAR(train_tensor)
    start = time.time()
    dct_ltar.fit(19, "dct")
    end = time.time()
    ltar.append(end-start)

    dct_lmlds = LMLDS(train_tensor)
    start = time.time()
    dct_lmlds.fit()
    end = time.time()
    lmlds.append(end-start)

    # dct_ltari = LTARI(train_tensor)
    # start = time.time()
    # dct_ltari.fit(19, 1, "dct")
    # end = time.time()
    # ltari.append(end-start)

    # interval = 24
    # dct_lstar = LSTAR(train_tensor)
    # start = time.time()
    # dct_lstar.fit(3, interval, "dct")
    # end = time.time()
    # lstar.append(end-start)

print("LSTM:", np.average(lstm))
print("LMLDS:", np.average(lmlds))
print("L-TAR:", np.average(ltar))
print("L-TARI:", np.average(ltari))
print("L-STAR:", np.average(lstar))

yhat = model.predict(test_X, verbose=0)
predict_tensor = yhat.reshape((N_test, tensor_shape[1], tensor_shape[2]))
dct_result_tensor = dct_ltar.forecast(N_test)
lmlds_result_tensor = dct_lmlds.forecast(N_test)
# dcti_result_tensor = dct_ltari.forecast(N_test)
# dcts_result_tensor = dct_lstar.forecast(N_test)

dct_error = jts.calc_mape_per_matrix(test_tensor, dct_result_tensor)
dct_error = dct_error.rename(columns={"MAPE": "dct-TAR"})
# dcti_error = jts.calc_mape_per_matrix(test_tensor, dcti_result_tensor)
# dcti_error = dcti_error.rename(columns={"MAPE": "dct-TARI"})
# dcts_error = jts.calc_mape_per_matrix(test_tensor, dcts_result_tensor)
# dcts_error = dcts_error.rename(columns={"MAPE": "dct-STAR"})
lstm_error = jts.calc_mape_per_matrix(test_tensor, predict_tensor)
lstm_error = lstm_error.rename(columns={"MAPE": "LSTM"})
mlds_error = jts.calc_mape_per_matrix(test_tensor, lmlds_result_tensor)
mlds_error = mlds_error.rename(columns={"MAPE": "dct-MLDS"})
df = pd.concat([mlds_error, dct_error, lstm_error], axis=1) #dcti_error, dcts_error, 
ax = df.plot()
ax.set_xlabel("Hours")
ax.set_ylabel("Error")
plt.show()