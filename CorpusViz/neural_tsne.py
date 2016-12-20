import numpy as np

from DCNN.util import rand_init, dotBias, tanh, tanh_prime
from CorpusViz.tsne import x2p, y2q, deltaY, kl_loss
import DCNN.gradient_step as gradient


def tsne(X, X_big=None, **kwargs):
    n, dim = X.shape
    model = NeuralTsne(dim)
    model.fit(X, **kwargs)

    if X_big is not None:
        Y = model.forward(X_big)
    else:
        Y = model.forward(X)

    return model, Y[-1]


class NeuralTsne(object):
    """docstring for NeuralTsne"""
    def __init__(self, input_dim, out_dim=2, n_layers=3, dims=None):
        super(NeuralTsne, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.n_layers = n_layers

        if dims is None:
            dims = range(input_dim, out_dim, (out_dim - input_dim) / n_layers)
            dims.append(out_dim)

        weights = []
        for i in range(n_layers):
            weights.append(rand_init(dims[i + 1], dims[i], bias=True))

        self.weights = weights

    def forward(self, X):
        a = [X.T]
        weights = self.weights
        n, dim = X.shape

        for i in range(self.n_layers):
            a.append(dotBias(weights[i], a[i]))
            if i + 1 < self.n_layers:
                a[i + 1] = tanh(a[i + 1])

        Y = a[-1].T

        num, Q = y2q(Y)

        return a, num, Q, Y

    def function_prime(self, X, P):

        a, num, Q, Y = self.forward(X)
        loss = kl_loss(P, Q)

        # Compute gradient
        Ws = self.weights
        n_layers = self.n_layers

        dY = 500 * deltaY(P, Y, num, Q)

        deltas = [None for i in range(n_layers + 1)]
        deltas[-1] = dY.T
        dWs = [None for i in range(n_layers)]

        for i in range(n_layers)[::-1]:
            deltas[i] = np.dot(Ws[i].T, deltas[i + 1])[:-1, :]
            deltas[i] *= tanh_prime(a[i])
            dWs[i] = np.zeros(Ws[i].shape)
            dWs[i][:, :-1] = np.dot(deltas[i + 1], a[i].T)
            dWs[i][:, -1] = np.sum(deltas[i + 1], axis=1)

        return loss, dWs

    def function_prime_all(self, X, P):

        a, num, Q, Y = self.forward(X)
        loss = kl_loss(P, Q)

        # Compute gradient
        Ws = self.weights
        n_layers = self.n_layers

        dY = deltaY(P, Y, num, Q)

        deltas = [None for i in range(n_layers + 1)]
        deltas[-1] = dY.T
        dWs = [None for i in range(n_layers)]

        for i in range(n_layers)[::-1]:
            deltas[i] = np.dot(Ws[i].T, deltas[i + 1])[:-1, :]
            deltas[i] *= tanh_prime(a[i])
            dWs[i] = np.zeros(Ws[i].shape)
            dWs[i][:, :-1] = np.dot(deltas[i + 1], a[i].T)
            dWs[i][:, -1] = np.sum(deltas[i + 1], axis=1)

        return loss, Q, dWs, dY

    def fit(self,
            X,
            lr=500,
            perplexity=30,
            epoch=1000,
            information_freq=10
            ):

        weights = self.weights
        # descender = gradient.Adadelta(500, 0.01, 0.9, 0.001, *weights)
        descender = gradient.Momentum(1.0, list_Ws=weights)
        n, dim = X.shape
        # mini_batch_size = n

        w0 = weights[0].copy()

        # Compute P-values
        P = x2p(X, 1e-5, perplexity)
        P = P * 4         # early exaggeration
        P = np.maximum(P, 1e-12)

        print 'Starting the gradient descent...', '\n'

        for j in range(epoch):

            loss, deltas = self.function_prime(X, P)
            descender.acc(deltas)
            descender.end_batch(1)

            w1 = weights[0].copy()
            dw0_1 = np.linalg.norm(w0 - w1)
            # print 'update norm:', dw0_1
            if dw0_1 < 1e-8:
                break
            w0 = w1

            if j == 100:
                print 'Stop lying about P-values'
                P /= 4

            if j % information_freq == 0:
                print 'Loss at epoch %i: %f' % (j, loss)
                descender.information_display()

        if (epoch - 1) % information_freq == 0:
            print 'Saving last model...'
            self.save_model('toto')
            print 'Saved!'

            print 'Loss at epoch %i: %f' % (j, loss)

    def gradient_check(self, e=1e-8, eps=1e-5, n=100):

        X = np.random.rand(n, self.input_dim)
        P = x2p(X, 1e-5, 30)
        P = np.maximum(P, 1e-12)

        Y = self.forward(X)[-1]
        loss, Q, dWs, dY = self.function_prime_all(X, P)

        err = 0

        err += self._gradient_check_Y(dY, Y, P, e, eps)

        weights = self.weights

        for i, dW in enumerate(dWs):
            assert dW.shape == weights[i].shape
            err += self._gradient_check_2(i, dW, X, P, e, eps)

        return err

    def _gradient_check_Y(self, dY, Y, P, e, eps):

        err = 0

        estimated_dY = np.zeros((10, Y.shape[1]))

        for j in range(Y.shape[1]):
            for i in range(10):
                Y[i, j] += e
                _, Q = y2q(Y)
                Jpos = kl_loss(P, Q)
                Y[i, j] -= 2 * e
                _, Q = y2q(Y)
                Jneg = kl_loss(P, Q)
                Y[i, j] += e

                dJ = (Jpos - Jneg) / (2*e)
                estimated_dY[i, j] = dJ
                backprop = float(dY[i, j])
                if abs(dJ - backprop) > eps:
                    print 'error on dY_%i_%i:' % (j, i), dJ, backprop
                    err += 1

        # print sigma.shape
        print estimated_dY[:, 0] / dY[:10, 0]
        return err

    def _gradient_check_2(self, name, dW, X, P, e, eps):
        W = self.weights[name]
        assert len(W.shape) == 2
        err = 0

        for j in range(W.shape[0]):
            for i in range(W.shape[1]):
                W[j, i] += e
                a, num, Q, Y = self.forward(X)
                Jpos = kl_loss(P, Q)
                W[j, i] -= 2 * e
                a, num, Q, Y = self.forward(X)
                Jneg = kl_loss(P, Q)
                W[j, i] += e

                dJ = (Jpos - Jneg) / (2*e)
                backprop = float(dW[j, i])
                if abs(dJ - backprop) > eps:
                    print 'error on d%s_%i_%i:' % (name, j, i), dJ, backprop
                    err += 1
        return err


def test_neural_tsne(input_dim=5):
    tsne = NeuralTsne(input_dim=5)
    for W in tsne.weights:
        print W.shape

    err = tsne.gradient_check()

    print 'found', err, 'errors...'
    assert err == 0


if __name__ == "__main__":

    test_neural_tsne()
