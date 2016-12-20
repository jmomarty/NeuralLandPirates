def gradient_check(loss_backward, loss, weights, X, y, e=1e-8, eps=1e-5):

    J, dW = loss_backward(X, y)

    err = 0

    for name, dW in dW.iteritems():
        W = weights[name]
        assert dW.shape == W.shape
        if len(dW.shape) == 2:
            err += _gradient_check_2(loss, name, W, dW, X, y, e, eps)
        elif len(dW.shape) == 3:
            err += _gradient_check_3(loss, name, W, dW, X, y, e, eps)
        elif len(dW.shape) == 4:
            err += _gradient_check_4(loss, name, W, dW, X, y, e, eps)

    return err


def _gradient_check_2(loss, name, W, dW, X, y, e, eps):
    assert len(W.shape) == 2
    err = 0

    for j in range(W.shape[0]):
        for i in range(W.shape[1]):
            W[j, i] += e
            Jpos = loss(X, y)
            W[j, i] -= 2 * e
            Jneg = loss(X, y)
            W[j, i] += e

            dJ = (Jpos - Jneg)/(2*e)
            backprop = float(dW[j, i])
            if abs(dJ - backprop) > eps:
                print 'error on d%s_%i_%i:' % (name, j, i), dJ, backprop
                err += 1
    return err


def _gradient_check_3(loss, name, W, dW, X, y, e, eps):
    assert len(W.shape) == 3
    err = 0
    for k in range(W.shape[0]):
        for j in range(W.shape[1]):
            for i in range(W.shape[2]):
                W[k, j, i] += e
                Jpos = loss(X, y)
                W[k, j, i] -= 2 * e
                Jneg = loss(X, y)

                dJ = (Jpos - Jneg)/(2*e)
                backprop = float(dW[k, j, i])
                if abs(dJ - backprop) > eps:
                    print 'error on d%s_%i_%i_%i:' % (name, k, j, i), dJ, backprop
                    err += 1
                W[k, j, i] += e
    return err


def _gradient_check_4(loss, name, W, dW, X, y, e, eps):
    assert len(W.shape) == 4
    err = 0
    for l in range(W.shape[0]):
        for k in range(W.shape[1]):
            for j in range(W.shape[2]):
                for i in range(W.shape[3]):
                    W[l, k, j, i] += e
                    Jpos = loss(X, y)
                    W[l, k, j, i] -= 2 * e
                    Jneg = loss(X, y)

                    dJ = (Jpos - Jneg)/(2*e)
                    backprop = float(dW[l, k, j, i])
                    if abs(dJ - backprop) > eps:
                        print 'error on d%s_%i_%i_%i_%i:' % (name, l, k, j, i), dJ, backprop
                        err += 1
                    W[l, k, j, i] += e
    return err
