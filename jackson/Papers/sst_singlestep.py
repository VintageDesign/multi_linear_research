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

train_tensor = jts.extract_train_tensor(tensor_data, N_train)
test_tensor = jts.extract_test_tensor(tensor_data, N_train, N_test)

ltar = []
for i in range(1):
    dct_ltar = LTAR(train_tensor)
    start = time.time()
    dct_ltar.fit(5, "dct")
    end = time.time()
    ltar.append(end-start)

print("L-TAR:", np.average(ltar))

dct_result_tensor = dct_ltar.single_step_forecast(N_test, test_tensor)

dct_error = jts.calc_mape_per_matrix(test_tensor, dct_result_tensor)
dct_error = dct_error.rename(columns={"MAPE": "dct-TAR"})
mlds_err = sio.loadmat('err_data\\err_sst_single.mat')
mlds_error = pd.DataFrame(np.transpose(mlds_err['err_dct']), index=dct_error.index, columns=["dct-MLDS"])
df = pd.concat([mlds_error, dct_error], axis=1)
ax = df.plot()
ax.set_xlabel("Hours")
ax.set_ylabel("Error")
plt.show()