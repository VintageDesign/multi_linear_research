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
N_test = N - N_train
print(f"N: {N}")
print(f"N_train: {N_train}")
print(f"N_test: {N_test}")

tensor_shape = (1171, 10, 10)

tensor_data = np.zeros(tensor_shape)
for i in range(tensor_shape[0]):
    tensor_data[i] = video['X'][0][i]

train_tensor = jts.extract_train_tensor(tensor_data, N_train)
test_tensor = jts.extract_test_tensor(tensor_data, N_train, N_test)


ltar = []
for i in range(1):
    dct_ltar = LTAR(train_tensor)
    start = time.time()
    dct_ltar.fit(10, "dwt")
    end = time.time()
    ltar.append(end-start)

print("L-TAR:", np.average(ltar))

dct_result_tensor = dct_ltar.single_step_forecast(N_test, test_tensor)

dct_error = jts.calc_mape_per_matrix(test_tensor, dct_result_tensor)
dct_error = dct_error.rename(columns={"MAPE": "dwt-TAR"})
mlds_err = sio.loadmat('err_data\\err_video_single.mat')
mlds_error = pd.DataFrame(np.transpose(mlds_err['err_dft']), index=dct_error.index, columns=["dwt-MLDS"])
df = pd.concat([mlds_error, dct_error], axis=1)
df = df[df.index < 160]
print(df)
ax = df.plot()
ax.set_xlabel("Frames")
ax.set_ylabel("Error")
plt.show()