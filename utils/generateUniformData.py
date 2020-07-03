import numpy as np


def generate_binary_distribution(batchsize, dim):
    z_batch = np.zeros((batchsize, dim))
    for b in range(batchsize):
        value_zeros_ones = np.zeros((dim))
        for i in range(dim):
            if i < dim // 2:
                value_zeros_ones[i] = 0.
            else:
                value_zeros_ones[i] = 1.
        index = np.arange(dim)
        np.random.shuffle(index)
        z_batch[b, ...] = value_zeros_ones[index]
    return z_batch
