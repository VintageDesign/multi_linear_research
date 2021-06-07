import scipy.io as sio
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

video = sio.loadmat('data/video.mat')

tensor_shape = (1171, 10, 10)
tensor_data = np.zeros(tensor_shape)
for i in range(tensor_shape[0]):
    tensor_data[i] = video['X'][0][i]

N = 1171
N_train = 1000
N_test = 171


def animate(i):
    im.set_array(tensor_data[i])
    return im,
fig = plt.figure()
im = plt.imshow(tensor_data[0], cmap=plt.get_cmap("gray"), animated = True)
ani = FuncAnimation(fig, animate, interval = 100, frames = N, blit=True)
plt.show()