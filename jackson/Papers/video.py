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
from ltar import LTAR, diff, invert_diff
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

import time

video = sio.loadmat("F:\\repos\multi_linear_research\jackson\Docs&Code\L-TVAR\data\\video.mat")

N = 1171
N_train = 1000
N_test = 171
print(f"N: {N}")
print(f"N_train: {N_train}")
print(f"N_test: {N_test}")

tensor_shape = (1171, 10, 10)

tensor_data = np.zeros(tensor_shape)
for i in range(tensor_shape[0]):
    tensor_data[i] = video['X'][0][i]

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
n_steps = 10
sequence = tensor_data.reshape((N, tensor_shape[1] * tensor_shape[2]))
X, y = split_sequence(sequence, n_steps)
train_X = X[:N_train-n_steps]
train_y = y[:N_train-n_steps]
test_X = X[N_train-n_steps:N_train+N_test-n_steps]
test_y = y[N_train-n_steps:N_train+N_test-n_steps]
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
model = Sequential()
model.add(LSTM(100, activation='relu',return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mse')

lstm = []
ltar = []
ltari = []
lstar = []
for i in range(1):
    start = time.time()
    model.fit(train_X, train_y, epochs=200, verbose=1)
    end = time.time()
    lstm.append(end-start)

    dct_ltar = LTAR(train_tensor)
    start = time.time()
    dct_ltar.fit(10, "dwt")
    end = time.time()
    ltar.append(end-start)

    dct_ltari = LTAR(diff(train_tensor))
    start = time.time()
    dct_ltari.fit(10, "dwt")
    end = time.time()
    ltari.append(end-start)

    interval = 10
    dct_lstar = LTAR(diff(train_tensor, interval))
    start = time.time()
    dct_lstar.fit(10, "dwt")
    end = time.time()
    lstar.append(end-start)

print("LSTM:", np.average(lstm))
print("L-TAR:", np.average(ltar))
print("L-TARI:", np.average(ltari))
print("L-STAR:", np.average(lstar))

yhat = model.predict(test_X, verbose=0)
predict_tensor = yhat.reshape((N_test, tensor_shape[1], tensor_shape[2]))
dct_result_tensor = dct_ltar.forecast(N_test)
dcti_result_tensor = invert_diff(dct_ltari.forecast(N_test), train_tensor)
dcts_result_tensor = invert_diff(dct_ltari.forecast(N_test), train_tensor, interval)

dct_error = jts.calc_mape_per_matrix(test_tensor, dct_result_tensor)
dct_error = dct_error.rename(columns={"MAPE": "dwt-TAR"})
dcti_error = jts.calc_mape_per_matrix(test_tensor, dcti_result_tensor)
dcti_error = dcti_error.rename(columns={"MAPE": "dwt-TARI"})
dcts_error = jts.calc_mape_per_matrix(test_tensor, dcts_result_tensor)
dcts_error = dcts_error.rename(columns={"MAPE": "dwt-STAR"})
lstm_error = jts.calc_mape_per_matrix(test_tensor, predict_tensor)
lstm_error = lstm_error.rename(columns={"MAPE": "LSTM"})
mlds_err = sio.loadmat('F:\\repos\multi_linear_research\jackson\Docs&Code\L-TVAR\data\err_video.mat')
mlds_error = pd.DataFrame(np.transpose(mlds_err['err_dft']), index=dct_error.index, columns=["dft-MLDS"])
df = pd.concat([mlds_error, lstm_error, dct_error, dcti_error, dcts_error], axis=1)
ax = df.plot()
ax.set_xlabel("Frames")
ax.set_ylabel("Error")
plt.show()