import numpy as np


def iteritems(d):
    for k, v in d.iteritems():
        if isinstance(v, list):
            for i, w in enumerate(v):
                yield k + '$' + str(i), w
        else:
            yield k, v


def wrap_weights(Ws):
    return dict(iteritems(Ws))


def unwrap_weights(Ws):
    d = {}
    to_list = set()

    for k, v in d.iteritems():
        if '$' in k:
            name, index = k.split('$', 1)
            to_list.add(name)
            if name not in d:
                d[name] = {}
            d[name][int(index)] = v

    for name in to_list:
        l = range(len(d[name]))
        for i, v in d[name].iteritems():
            l[i] = v
        d[name] = l

    return d


def map_values(f, dict1):
    return {k: f(v) for k, v in iteritems(dict1)}


class GradientHelper(object):

    def __init__(self, reg=0, **Ws):
        self.reg = reg

        self.Ws = dict(iteritems(Ws))
        self.dWs = map_values(lambda W: np.zeros(W.shape), self.Ws)

        self.batch_size = 0

    def acc(self, **dWs):
        self.batch_size += 1
        for k, dW in iteritems(dWs):
            self.dWs[k] += dW

    def begin_batch(self):
        self.batch_size = 0

        for k, dW in self.dWs.iteritems():
            dW *= 0

    def end_batch(self):
        n = self.batch_size

        for k, dW in self.dWs.iteritems():
            dW /= n
            if self.reg > 0:
                dW += (self.reg / self.Ws[k].size) * self.Ws[k]

        self.gradient_step()

    def gradient_step(self):
        raise NotImplemented

    def information_display(self):
        print 'lr:', self.lr
        for n, lr in self.learning_rates():
            print 'learning rate for layer %s:\t min %f,\t max %f, \t mean %f' % \
                (n, np.min(lr), np.max(lr), np.mean(lr))


class ClassicGrad(GradientHelper):

    def __init__(self, lr, reg=0.01, n0=10, **Ws):
        GradientHelper.__init__(self, reg, **Ws)
        self.lr = lr * np.sqrt(n0)
        self.updates = n0

    def gradient_step(self):
        for k, W in self.Ws.iteritems():
            dW = self.dWs[k]
            alpha = np.linalg.norm(dW)
            self.updates += 1
            W -= self.lr / np.sqrt(self.updates) / alpha * dW

    def learning_rates(self):
        lr = self.lr / np.sqrt(self.updates)
        for k in self.Ws.iterkeys():
            yield k, lr

    def reset(self):
        for h in self.hist.values():
            h *= 0
            h += self.eps


class Adagrad(GradientHelper):

    def __init__(self, lr, reg=0.01, eps=1e-10, **Ws):
        GradientHelper.__init__(self, reg, **Ws)
        self.hist = {}
        self.lr = lr
        self.eps = eps

        for k, W in self.Ws.iteritems():
            self.hist[k] = np.zeros(W.shape) + eps

    def gradient_step(self):
        hist = self.hist
        for k, W in self.Ws.iteritems():
            dW = self.dWs[k]
            hist[k] += dW * dW
            dW /= np.sqrt(hist[k])
            W -= self.lr * dW

    def learning_rates(self):
        for k, hist in self.hist.iteritems():
            yield k, 1.0 / np.sqrt(hist)

    def reset(self):
        for h in self.hist.values():
            h *= 0
            h += self.eps


class Adadelta(GradientHelper):

    def __init__(self, lr, reg=0.01, rho=0.9, eps=1e-10, **Ws):
        GradientHelper.__init__(self, reg, **Ws)

        self.hist = {}
        self.rho = rho
        self.eps = eps
        self.lr = lr
        self._lr = {}

        for k, W in self.Ws.iteritems():
            self.hist[k] = np.zeros(W.shape), np.zeros(W.shape)


    def gradient_step(self):
        eps = self.eps
        for k, W in self.Ws.iteritems():
            dW = self.dWs[k]
            hist, hist2 = self.hist[k]

            hist *= self.rho
            hist += (1 - self.rho) * np.square(dW)

            lr = np.sqrt((hist2 + eps) / (hist + eps))
            self._lr[k] = lr
            dW *= lr
            W -= self.lr * dW

            hist2 *= self.rho
            hist2 += (1 - self.rho) * np.square(dW)

    def learning_rates(self):
        return self._lr.iteritems()

    def reset(self):
        for h, h2 in self.hist.values():
            h *= 0
            h2 *= 0


class RMSprop(GradientHelper):
    def __init__(self, rho_up=1.2, rho=0.9, eps=0.001, **Ws):
        GradientHelper.__init__(self, **Ws)

        self.hist = {}
        self.rho_up = rho_up
        self.rho = rho
        self.eps = eps

        for (k, W) in self.Ws.iteritems():
            self.hist[k] = np.zeros(W.shape), np.ones(W.shape), np.zeros(W.shape)

    def gradient_step(self):
        for k, W in self.Ws.iteritems():
            dW = self.dWs[k]
            hist, momentum, prev = self.hist[k]

            hist *= self.rho
            hist += (1 - self.rho) * dW * dW

            momentum *= (self.rho_up - 0.5) * (prev * dW >= 0) + .5
            momentum.clip(1e-6, 50, out=momentum)

            dW *= momentum / np.sqrt(hist + self.eps)

            W -= dW

            prev *= 0
            prev += np.sign(dW)

    def reset(self):
        for hist, momentum, prev in self.hist.itervalues():
            hist *= 0
            momentum *= 0
            momentum += 1
            prev *= 0

    def learning_rates(self):
        for k, (hist, momentum, prev) in self.hist.iteritems():
            yield k, momentum / np.sqrt(hist + self.eps)


class Momentum(GradientHelper):
    def __init__(self, lr, reg=0.0,
                 rho_init=0.5, rho=0.8,
                 min_gain=0.01, max_gain=10.0,
                 **Ws
                 ):
        GradientHelper.__init__(self, reg, **Ws)

        self.lr = lr
        self.rho_init = rho_init
        self.rho = rho
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.epoch = 0

        self.hist = {}
        for (k, W) in self.Ws.iteritems():
            self.hist[k] = np.zeros(W.shape), np.zeros(W.shape)

    def gradient_step(self):
        for k, W in self.Ws.iteritems():
            rho = self.rho_init if self.epoch < 20 else self.rho

            dW = self.dWs[k]
            gains, iW = self.hist[k]

            gains = (gains + 0.2) * ((dW > 0) == (iW > 0)) + (gains * 0.8) * ((dW > 0) != (iW > 0))
            gains = np.maximum(gains, self.min_gain)
            gains = np.minimum(gains, self.max_gain)

            iW = rho * iW + self.lr * (gains * dW)
            W -= iW

            self.hist[k] = gains, iW
        self.epoch += 1

    def reset(self):
        for gains, iW in self.hist.itervalues():
            gains *= 0
            iW *= 0
        self.epoch = 0

    def learning_rates(self):
        for k, (gains, iW) in self.hist.iteritems():
            yield k, iW


class Adam(GradientHelper):

    def __init__(self, lr, reg=0, beta1=0.9, beta2=0.99, eps=1e-8, **Ws):
        GradientHelper.__init__(self, reg, **Ws)
        self.smoothed_grad = {}
        self.momentum = {}
        self.time = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.eps = eps

        for k, W in self.Ws.iteritems():
            self.smoothed_grad[k] = np.zeros(W.shape)
            self.momentum[k] = np.zeros(W.shape)

    def gradient_step(self):
        self.time += 1
        time = self.time
        beta1 = self.beta1
        beta2 = self.beta2

        for k, W in self.Ws.iteritems():
            dW = self.dWs[k]

            smoothed = self.smoothed_grad[k]
            smoothed *= beta1
            smoothed += (1 - beta1) * dW

            momentum = self.momentum[k]
            momentum *= beta2
            momentum += (1 - beta2) * dW * dW

            time_correction = np.sqrt(1 - np.power(beta1, time)) / (1 - np.power(beta2, time))

            W -= (self.lr * time_correction) * smoothed / (np.sqrt(momentum) + self.eps)

    def learning_rates(self):
        time = self.time
        beta1 = self.beta1
        beta2 = self.beta2

        for k in self.Ws.iterkeys():
            smoothed = self.smoothed_grad[k]
            momentum = self.momentum[k]
            time_correction = np.sqrt(1 - np.power(beta1, time)) / (1 - np.power(beta2, time))
            yield k, (self.lr * time_correction) / (np.sqrt(momentum) + self.eps)

    def reset(self):
        for k in self.Ws.keys():
            self.smoothed_grad[k] *= 0
            self.momentum[k] *= 0
