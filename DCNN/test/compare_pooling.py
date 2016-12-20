import numpy as np

from DCNN.convolve import pool, unpool


def test_pool_4_100_10_5():
    d1 = 4
    d2 = 100
    d3 = 10
    kmax = 5

    return helper_pool(d1, d2, d3, kmax)


def test_pool_6_50_5_3():
    d1 = 6
    d2 = 50
    d3 = 5
    kmax = 3

    return helper_pool(d1, d2, d3, kmax)


def test_unpool_4_100_10_5():
    d1 = 4
    d2 = 100
    d3 = 10
    kmax = 5

    return helper_unpool(d1, d2, d3, kmax)


def test_unpool_6_50_5_3():
    d1 = 6
    d2 = 50
    d3 = 5
    kmax = 3

    return helper_unpool(d1, d2, d3, kmax)


def test_bias():
    from DCNN.util import dotBias, rand_init
    d_in = np.random.randint(10, 100)
    d_out = np.random.randint(5, d_in)
    M = rand_init(d_out, d_in, bias=True)
    y = np.random.random(d_in)

    my_1 = M.dot(np.concatenate((y, [1])))
    my_2 = dotBias(M, y)

    print 'expected:', my_1.shape, ', got:', my_2.shape
    assert my_1.shape == my_2.shape
    assert np.linalg.norm(my_1 - my_2) < 1e-6


def helper_pool(d1, d2, d3, kmax):
    c = np.random.randint(-10, 10, (d1, d2, d3))
    z1, i1 = pool_v1(c, kmax)
    z2, i2 = pool(c, kmax)

    assert np.sum(z1 != z2) == 0


def helper_unpool(d1, d2, d3, kmax):
    c = np.random.randint(-10, 10, (d1, d2, d3))
    z, i1 = pool_v1(c, kmax)
    z, i2 = pool(c, kmax)

    d1 = unpool_v1(z, i1, d3)
    d2 = unpool(z, i2, d3)

    assert np.sum(d1 != d2) == 0


def pool_v1(c, kmax):

    d1, d2, d3 = c.shape
    z = np.zeros((d1, d2, kmax))

    kmax_indexes = {}
    for i in range(d1):
        kmax_indexes[i] = np.zeros((d2, d3))

    # Kmax-Pooling
    for i in range(d1):
        kmax_indexes[i] = c[i].argsort()[:, -kmax:]
        kmax_indexes[i].sort()
        for j in range(d2):
            z[i][j] = c[i][j][kmax_indexes[i][j]]

    return z, kmax_indexes


def unpool_v1(delta, indexes, length):
    d1, d2, kmax = delta.shape
    delta_after = np.zeros((d1, d2, length))

    # Through K-Max Pooling Layer:
    for i in range(d1):
        for k in range(d2):
            for p in range(kmax):
                delta_after[i][k, indexes[i][k, p]] = delta[i][k, p]

    return delta_after


if __name__ == '__main__':
    print '--- simulating 2nd layer pooling ---'
    print 'forward'
    test_pool_6_50_5_3()

    print 'backward'
    test_unpool_6_50_5_3()

    print '------------'
    print 'simulating 1st convolution layer'

    print 'forward'
    test_pool_4_100_10_5()

    print 'backward'
    test_unpool_4_100_10_5()

    print 'everything is okay !'
