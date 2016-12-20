import numpy as np


def tanh(z):
    return np.tanh(z)


def tanh_prime(fz):
    return 1 - fz ** 2


def softmax(z):
    y = np.exp(z)
    return y/sum(y)


def ReLU(z):
    return np.maximum(z, 0)


def ReLU_prime(fz):
    return fz != 0


def rand_init(l_out, l_in, bias=False):
    var = np.sqrt(6.0/(l_out+l_in))
    W = np.random.uniform(-var, var, (l_out, l_in + bias))
    W[:, -1] = 0
    return W


def dotBias(W, x):
    """
        Matrix multiplication + add a bias
        :param W: shape = (i, j + 1), bias is last column
        :param x: shape = (j) or (j, k)
        :return: W.x + b
    """
    return W[:, :-1].dot(x) + W[:, -1]
