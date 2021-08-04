import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.insert(0, '../jackson/Libraries')
import JacksonsTSPackage as jts
from ltar import LTAR, diff, invert_diff

#N = 39868
N = 2658

data_tensor = np.loadtxt("Data/baboon_tensor_15min.txt", delimiter=",").reshape((N, 13, 13))
print(data_tensor)
print(data_tensor.shape)

def plot_norm(tensor, N):
    norms = []
    for i in range(N):
        norms.append(np.linalg.norm(tensor[i], ord="fro"))
    pd.DataFrame(norms).plot()
    plt.show()
plot_norm(data_tensor, len(data_tensor))

N_train = 2400
N_test = N - N_train

train_tensor = jts.extract_train_tensor(data_tensor, N_train)
test_tensor = jts.extract_test_tensor(data_tensor, N_train, N_test)

ltar = LTAR(diff(train_tensor, 150))
ltar.fit(5)
forecast_tensor =  invert_diff(ltar.forecast(N_test), train_tensor, interval = 150)

norms = pd.DataFrame(columns=["Test", "Forecast"])
for i in range(N_test):
    df = pd.DataFrame([[np.linalg.norm(test_tensor[i], ord="fro"), np.linalg.norm(forecast_tensor[i], ord="fro")]], columns=["Test", "Forecast"])
    norms = norms.append(df, ignore_index=True)

norms.plot()
plt.show()

ltar_error = jts.calc_mape_per_matrix(test_tensor, forecast_tensor)
ltar_error.plot()
plt.show()