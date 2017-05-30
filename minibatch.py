import numpy as np
from PIL import Image

WIDTH = 64
HEIGHT = 64
C = 24

from scipy import misc


def get_data(data, labels):
    n = len(data)
    X = np.zeros((n, WIDTH, HEIGHT, 3), dtype = np.float32)
    y = np.zeros((n, C), dtype = np.float32)

    for i in range(n):
        img_array = misc.imread(data[i])
        img_array.astype(dtype=np.float32, copy=False)

        # zero centered image
        img_array = img_array - img_array.mean(axis=(0, 1))
        X[i] = img_array
        y[i] = labels[i]


    return X, y


def get_batch(data, label, batch_number, batch_size=128):
    # return X, y with X resized images and y labels
    min_index = batch_number * batch_size
    max_index = min(len(data), (batch_number + 1) * batch_size)
    return get_data(data[min_index: max_index], label[min_index: max_index])

def shuffle_data(data, labels):
    indices = np.arange(len(data))
    np.random.shuffle(indices)

    return data[indices], labels[indices]