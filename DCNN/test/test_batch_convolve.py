import numpy as np

import DCNN.convolve as cv


def test_batch_convolve(fmap2=6, fmap1=4, kernel_size=5, dim=30, length=40, batch=28):
    kernel = np.random.normal(0, 0.5, (fmap2, fmap1, dim, kernel_size))
    sentences = np.random.normal(0, 0.5, (fmap1, dim, length, batch))

    out_1 = np.zeros((fmap2, dim, length + kernel_size - 1, batch))
    for b in range(batch):
        out_1[:, :, :, b] = cv.convolution(kernel, sentences[:, :, :, b])

    out_2 = cv.convolve_batch(kernel, sentences)

    assert out_1.shape == out_2.shape

    diff = np.linalg.norm(out_1 - out_2)
    assert diff < 1e-6


def test_batch_correlate(fmap2=6, fmap1=4, kernel_size=5, dim=30, length=40, batch=28):
    kernel = np.random.normal(0, 0.5, (fmap2, fmap1, dim, kernel_size))
    sentences = np.random.normal(0, 0.5, (fmap2, dim, length, batch))

    out_1 = np.zeros((fmap1, dim, length - kernel_size + 1, batch))
    for b in range(batch):
        out_1[:, :, :, b] = cv.correlate(kernel, sentences[:, :, :, b])

    out_2 = cv.correlate_batch(kernel, sentences)

    assert out_1.shape == out_2.shape

    diff = np.linalg.norm(out_1 - out_2)
    assert diff < 1e-6


def test_batch_pool_unpool(fmap=4, dim=30, kmax=30, length=40, batch=28):
    sentences = np.random.normal(0, 0.5, (fmap, dim, length, batch))

    pooled_1 = np.zeros((fmap, dim, kmax, batch))
    indexes_1 = {}
    for b in range(batch):
        pooled_1[:, :, :, b], indexes_1[b] = cv.pool(sentences[:, :, :, b], kmax)

    pooled_2, indexes_2 = cv.pool_batch(sentences, kmax)

    assert pooled_1.shape == pooled_2.shape
    diff = np.sum(pooled_1 != pooled_2)
    assert diff == 0

    unpooled_1 = np.zeros((fmap, dim, length, batch))
    for b in range(batch):
        unpooled_1[:, :, :, b] = cv.unpool(pooled_1[:, :, :, b], indexes_1[b], length)

    unpooled_2 = cv.unpool_batch(pooled_2, indexes_2, length)
    assert unpooled_1.shape == unpooled_2.shape
    diff = np.sum(unpooled_1 != unpooled_2)
    assert diff == 0


if __name__ == '__main__':
    n = 10

    for i in range(n):
        test_batch_convolve()
        test_batch_correlate()
        test_batch_pool_unpool()

    print n, 'tests passed !'
