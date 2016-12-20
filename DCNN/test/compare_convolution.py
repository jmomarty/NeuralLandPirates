import numpy as np
import time

import DCNN.convolve as cv2


def rand_init(l_in, l_out):
    var = np.sqrt(6.0 / (l_in + l_out))
    return np.random.uniform(-var, var, (l_in, l_out))


def init_kernel(fmap2, fmap1, vector_size, kernel_size):
    CM = np.zeros((fmap2, fmap1, vector_size, kernel_size))
    for i in range(fmap2):
        for j in range(fmap1):
            CM[i, j] = rand_init(vector_size, kernel_size)

    return CM


def convolution_v1(CM2, a1):
    t0 = time.time()

    fmap2, fmap1, vector_size, kernel_size = CM2.shape
    fmap1_bis, vector_size_bis, length_before = a1.shape

    # assert fmap1_bis == fmap1
    # assert vector_size_bis == vector_size

    c2 = np.zeros((fmap2, vector_size, kernel_size + length_before - 1))

    for i in range(fmap2):
        for j in range(fmap1):
            for d in range(vector_size):
                c2[i, d] += np.convolve(a1[j, d], CM2[i, j, d])

    t1 = time.time()

    return c2, t1 - t0


def convolution_v2(CM2, a1):

    t0 = time.time()
    c2 = cv2.convolution(CM2, a1)
    t1 = time.time()

    return c2, t1 - t0


def correlate_v1(CM, after):

    t0 = time.time()
    fmap2, fmap1, vector_size, kernel_size = CM.shape
    fmap2_bis, vector_size_bis, length_after = after.shape
    length_before = length_after + 1 - kernel_size

    # assert fmap2_bis == fmap2
    # assert vector_size_bis == vector_size

    before = np.zeros((fmap1, vector_size, length_before))

    for j in range(fmap1):
        for i in range(fmap2):
            for d in range(vector_size):
                before[j, d] += np.correlate(after[i, d], CM[i, j, d], mode='valid')
    t1 = time.time()

    return before, t1 - t0


def correlate_v2(CM, after):

    t0 = time.time()
    before = cv2.correlate(CM, after)
    t1 = time.time()

    return before, t1 - t0


def kernel_gradient_v1(delta, a1):

    t0 = time.time()
    fmap2, vector_size, length = delta.shape
    fmap1, vector_size_bis, length_before = a1.shape
    kernel_size = length + 1 - length_before

    # assert vector_size_bis == vector_size
    dCM = np.zeros((fmap2, fmap1, vector_size, kernel_size))

    for i in range(fmap2):
        for j in range(fmap1):
            for d in range(vector_size):
                dCM[i, j, d] = np.correlate(delta[i, d], a1[j, d], mode='valid')

    t1 = time.time()

    return dCM, t1 - t0


def kernel_gradient_v2(delta, a1):

    t0 = time.time()
    dCM = cv2.kernel_gradient(delta, a1)
    t1 = time.time()

    return dCM, t1 - t0


def test_conv_corr_grad_layer1():
    print 'simulating 1st convolution layer'
    fmap2 = 4
    fmap1 = 3
    vector_size = 100
    kernel_size = 6
    length = 40
    eps = 1e-10

    helper_conv_corr_grad(fmap2, fmap1, vector_size, kernel_size, length, eps)


def test_conv_corr_grad_layer2():
    print '--- simulating 2nd convolution layer ---'
    fmap2 = 4
    fmap1 = 3
    vector_size = 50
    kernel_size = 5
    kmax = 3
    length = int(np.max([kmax, np.ceil(0.5 * 40)]))
    eps = 1e-10

    helper_conv_corr_grad(fmap2, fmap1, vector_size, kernel_size, length, eps)


def helper_conv_corr_grad(fmap2, fmap1, vector_size, kernel_size, length, eps):
    CM = init_kernel(fmap2, fmap1, vector_size, kernel_size)
    a = np.random.uniform(-1, 1, (fmap1, vector_size, length))

    print '--- simulating forward ---'
    c1, t1 = convolution_v1(CM, a)
    print 'method 1: time =', t1

    c2, t2 = convolution_v2(CM, a)
    print 'method 2: time =', t2
    print 'speed up = *%3f' % (t1 / t2)

    delta = np.linalg.norm(c1 - c2)
    print 'difference on result =', delta
    assert delta < eps

    print '--- simulating backprop ---'

    b1, t1 = correlate_v1(CM, c1)
    print 'method 1: time =', t1

    b2, t2 = correlate_v2(CM, c2)
    print 'method 2: time =', t2
    print 'speed up = *%3f' % (t1 / t2)
    delta = np.linalg.norm(b1 - b2)
    print 'difference on result =', delta
    assert delta < eps

    print '--- simulating kernel gradient ---'

    dCM1, t1 = kernel_gradient_v1(c1, a)
    print 'method 1: time =', t1

    dCM2, t2 = kernel_gradient_v2(c2, a)
    print 'method 2: time =', t2
    print 'speed up = *%3f' % (t1 / t2)

    delta = np.linalg.norm(dCM1 - dCM2)
    print 'difference on result =', delta
    assert delta < eps


def speed_up_stat_by_length(fmap2, fmap1, vector_size, kernel_size):
    CM = init_kernel(fmap2, fmap1, vector_size, kernel_size)

    for l in range(10, 100):
        a = np.random.uniform(-1, 1, (fmap1, vector_size, l))
        tt1, tt2 = 0, 0

        for k in range(50):
            c1, t1 = convolution_v1(CM, a)
            c2, t2 = convolution_v2(CM, a)
            tt1 += t1
            tt2 += t2
        s = tt1 / tt2
        print l, '->', s
        yield s


def speed_up_stat_by_kernel_size(fmap2, fmap1, vector_size, length):

    for kernel_size in range(1, 12):
        CM = init_kernel(fmap2, fmap1, vector_size, kernel_size)
        a = np.random.uniform(-1, 1, (fmap1, vector_size, length))
        tt1, tt2 = 0, 0

        for k in range(50):
            c1, t1 = convolution_v1(CM, a)
            c2, t2 = convolution_v2(CM, a)
            tt1 += t1
            tt2 += t2
        s = tt1 / tt2
        print kernel_size, '->', s
        yield s


def speed_up_stat(fmap2, fmap1, vector_size):

    stat = np.zeros((12, 100))

    for kernel_size in range(1, 12):
        for length in range(10, 100):
            CM = init_kernel(fmap2, fmap1, vector_size, kernel_size)
            a = np.random.uniform(-1, 1, (fmap1, vector_size, length))
            tt1, tt2 = 0, 0

            for k in range(10):
                c1, t1 = convolution_v1(CM, a)
                c2, t2 = convolution_v2(CM, a)
                tt1 += t1
                tt2 += t2
            s = tt1 / tt2
            print kernel_size, length, '->', s
            stat[kernel_size, length] = s
    return stat


def plot(s, x, y):
    s = list(s)

    import matplotlib.pyplot as plt
    plt.plot(s)
    plt.plot(map(lambda x: 1, s))
    plt.ylabel('speed up')
    plt.xlabel('sentence length')
    plt.show()


def plot2D(s, x, y):
    import matplotlib.pyplot as plt
    print s
    plt.imshow(s, interpolation='nearest')
    plt.show()


if __name__ == '__main__':
    print 'Simulating layer 2'
    test_conv_corr_grad_layer2()

    print '------------'
    print

    print 'Simulating layer 1'
    test_conv_corr_grad_layer1()

    fmap2 = 4
    fmap1 = 3
    vector_size = 50
    kernel_size = 5

    # s = speed_up_stat_by_kernel_size(fmap2, fmap1, vector_size, 60)
    # plot(s, x='sentence length', y='speed up')
    # s = speed_up_stat_by_length(fmap2, fmap1, vector_size, kernel_size)
    # plot(s, x='kernel_size', y='speed up')
    # s = speed_up_stat(4, 3, 50)
    # plot2D(s, 'kernel size', 'length')
