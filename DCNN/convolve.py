import numpy as np


def convolution(kernel, M):
    """
        Wide convolution on a sentence matrix
        :param kernel: shape = (fmap_out, fmap_in, d, kernel_size)
        :param M: sentence matrix, shape = (fmap_in, d, length)
        :return: new sentence matrix (fmap_out, d, kernel_size + length - 1)
    """
    fmap2, fmap1, vector_size, kernel_size = kernel.shape
    fmap1_bis, vector_size_bis, length = M.shape

    # assert fmap1_bis == fmap1
    # assert vector_size_bis == vector_size

    c2 = np.zeros((fmap2, vector_size, kernel_size + length - 1))
    for n in range(kernel_size + length - 1):
        for k in range(kernel_size):
            l = n - k
            if 0 <= l < length:
                c2[:, :, n] += np.sum(kernel[:, :, :, k] * M[:, :, l], axis=1)

    return c2


def convolve_batch(kernel, M):
    """
        Wide convolution on a sentence matrix
        :param kernel: shape = (fmap_out, fmap_in, d, kernel_size)
        :param M: sentence matrix, shape = (fmap_in, d, length, batch)
        :return: new sentence matrix (fmap_out, d, kernel_size + length - 1, batch)
    """
    fmap2, fmap1, vector_size, kernel_size = kernel.shape
    fmap1_bis, vector_size_bis, length, batch = M.shape

    # assert fmap1_bis == fmap1
    # assert vector_size_bis == vector_size

    c2 = np.zeros((fmap2, vector_size, kernel_size + length - 1, batch))
    for n in range(kernel_size + length - 1):
        for k in range(kernel_size):
            l = n - k
            if 0 <= l < length:
                c2[:, :, n, :] += _convolve_einsum(kernel[:, :, :, k], M[:, :, l, :])

    return c2


def _convolve_einsum(k, M):
    return np.einsum('fgd,gdb->fdb', k, M)


def correlate(kernel, delta):
    """
        Propagate the error through a convolution.
        :param kernel: shape = (fmap_out, fmap_in, d, kernel_size)
        :param delta: error after the layer, shape = (fmap_out, d, length_after)
        :return: error before convolution (fmap_in, d, length_before)
    """

    fmap2, fmap1, vector_size, kernel_size = kernel.shape
    fmap2_bis, vector_size_bis, length = delta.shape
    kmax1 = length + 1 - kernel_size

    before = np.zeros((fmap1, vector_size, kmax1))
    # assert fmap2_bis == fmap2
    # assert vector_size_bis == vector_size

    kernel = np.transpose(kernel, axes=[1, 0, 2, 3])

    for n in range(kmax1):
        for k in range(kernel_size):
            l = n + k
            if 0 <= l < length:
                before[:, :, n] += np.sum(kernel[:, :, :, k] * delta[:, :, l], axis=1)

    return before


def correlate_batch(kernel, delta):
    """
        Propagate the error through a convolution.
        :param kernel: shape = (fmap_out, fmap_in, d, kernel_size, batch)
        :param delta: error after the layer, shape = (fmap_out, d, length_after, batch)
        :return: error before convolution (fmap_in, d, length_before)
    """

    fmap2, fmap1, vector_size, kernel_size = kernel.shape
    fmap2_bis, vector_size_bis, length, batch = delta.shape
    kmax1 = length + 1 - kernel_size

    before = np.zeros((fmap1, vector_size, kmax1, batch))
    # assert fmap2_bis == fmap2
    # assert vector_size_bis == vector_size

    for n in range(kmax1):
        for k in range(kernel_size):
            l = n + k
            if 0 <= l < length:
                before[:, :, n, :] += _correlate_einsum(kernel[:, :, :, k], delta[:, :, l, :])

    return before


def _correlate_einsum(k, delta):
    return np.einsum('fgd,fdb->gdb', k, delta)


def kernel_gradient(delta, M):
    """
        Compute the gradient for the kernel
        :param delta: shape = (fmap_out, d, length_after)
        :param M: sentence matrix, shape = (fmap_in, d, length_before)
        :return: error for the kernel (fmap_out, fmap_in, d, kernel_size)
    """

    fmap2, vector_size, length = delta.shape
    fmap1, vector_size_bis, kmax = M.shape
    kernel_size = length + 1 - kmax

    # assert vector_size_bis == vector_size
    dCM = np.zeros((fmap2, fmap1, vector_size, kernel_size))

    for n in range(length):
        for k in range(kernel_size):
            l = n - k
            if 0 <= l < kmax:
                # dCM[:, :, :, k] += np.einsum('id,jd->jid', M[:, :, l], delta[:, :, n])
                dCM[:, :, :, k] += _kernel_einsum(M[:, :, l], delta[:, :, n])

    return dCM


def _kernel_einsum(a, b):
    return np.einsum('id,jd->jid', a, b)


def kernel_gradient_batch(delta, M):
    """
        Compute the gradient for the kernel
        :param delta: shape = (fmap_out, d, length_after, batch)
        :param M: sentence matrix, shape = (fmap_in, d, length_before, batch)
        :return: error for the kernel (fmap_out, fmap_in, d, kernel_size)
    """

    fmap2, vector_size, length, batch = delta.shape
    fmap1, vector_size_bis, kmax, batch_bis = M.shape
    kernel_size = length + 1 - kmax

    # assert batch = batch_bis
    # assert vector_size_bis == vector_size
    dCM = np.zeros((fmap2, fmap1, vector_size, kernel_size))

    for n in range(length):
        for k in range(kernel_size):
            l = n - k
            if 0 <= l < kmax:
                dCM[:, :, :, k] += np.einsum('idb,jdb->jid', M[:, :, l, :], delta[:, :, n, :])
                # dCM[:, :, :, k] += _kernel_einsum(M[:, :, l], delta[:, :, n])

    dCM /= batch
    return dCM


def pool(M, kmax, off=None):
    """
        Extract the kmax features of each dim in the sentence matrix
        :param M: sentence matrix, shape = (fmap, d, length)
        :param kmax: the number of features to pool
        :param off: a matrix with coordinates offsets
        :return:
            M_pooled: pooled sentence matrix, shape = (fmap, d, kmax)
            indexes: indexes of the pulled value, shape = (fmap * d * kmax)
    """
    d1, d2, d3 = M.shape

    if off is None:
        off = np.arange(d1 * d2).reshape(d1, d2, 1)

    indexes = M.argsort()[:, :, -kmax:]
    indexes.sort()
    indexes += d3 * off

    M_pooled = np.take(M, indexes)
    return M_pooled, indexes.reshape(-1)


def pool_batch(M, kmax, off=None):
    """
        Extract the kmax features of each dim in the sentence matrix
        :param M: sentence matrix, shape = (fmap, d, length, batch)
        :param kmax: the number of features to pool
        :param off: a matrix with coordinates offsets
        :return:
            M_pooled: pooled sentence matrix, shape = (fmap, d, kmax, batch)
            indexes: indexes of the pulled value, shape = (fmap * d * kmax * batch)
    """
    M = np.transpose(M, axes=[0, 1, 3, 2])
    d1, d2, d3, length = M.shape

    if off is None:
        off = np.arange(d1 * d2 * d3).reshape(d1, d2, d3, 1)

    indexes = M.argsort()[:, :, :, -kmax:]
    indexes.sort()
    indexes += length * off

    M_pooled = np.transpose(np.take(M, indexes), axes=[0, 1, 3, 2])
    return M_pooled, indexes.reshape(-1)


def unpool(delta, indexes, length):
    """
        Propagate the delta through the pooling
        :param delta: error on pooled sentence matrix, shape = (fmap, d, kmax)
        :param indexes: the indexes used for pooling
        :param length: the original length of the sentence matrix
        :return: the delta on orginal sentence matrix, shape = (fmap, d, length)
    """
    d1, d2, kmax = delta.shape
    delta_after = np.zeros(d1 * d2 * length)

    # Through K-Max Pooling Layer:
    delta_after[indexes] = delta
    delta_after = delta_after.reshape(d1, d2, length)

    return delta_after


def unpool_batch(delta, indexes, length):
    """
        Propagate the delta through the pooling
        :param delta: error on pooled sentence matrix, shape = (fmap, d, kmax, batch)
        :param indexes: the indexes used for pooling
        :param length: the original length of the sentence matrix
        :return: the delta on orginal sentence matrix, shape = (fmap, d, length, batch)
    """
    d1, d2, kmax, d3 = delta.shape
    delta_after = np.zeros(d1 * d2 * d3 * length)

    # Through K-Max Pooling Layer:
    delta_after[indexes] = np.transpose(delta, axes=[0, 1, 3, 2])
    delta_after = delta_after.reshape(d1, d2, d3, length)
    return np.transpose(delta_after, axes=[0, 1, 3, 2])
