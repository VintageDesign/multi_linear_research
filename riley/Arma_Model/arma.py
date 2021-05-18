from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

#from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def generate_data(t=1000):
    """
    Generate a random walk of size t
    """
    if t < 2:
        print("Error: t must be larger than 2")
        return 0

    sns.set()
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (14, 8)

    points = np.random.standard_normal(t)
    points[0] = 0

    random_walk = np.cumsum(points)

    return random_walk

random_walk = generate_data()
random_walk_series = pd.Series(random_walk)

plt.figure(figsize=[10, 7.5]); # Set dimensions for figure
plt.plot(random_walk)
plt.title("Simulated Random Walk")

random_walk_acf = acf(random_walk)

rw_diff = acf(np.diff(random_walk))

acf_plot = plot_acf(rw_diff, lags=20)

plt.show()



