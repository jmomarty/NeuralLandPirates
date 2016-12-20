# -*- coding: utf-8 -*-
import numpy as np
import datetime
import time

from gradient_step import Adadelta, Adagrad, RMSprop


class AbstractCNN(object):

    def __init__(self,
                 kmax=3,
                 num_labels=15,
                 vector_size=30
                 ):

        # Word2Vec Size
        self.vector_size = vector_size

        # Useful Size Variables
        self.kmax = kmax
        self.num_labels = num_labels
        self.weights = {}

    def sumsqr(self, a):
        return np.sum(a ** 2)

    def initialize_weights(self):
        raise Exception('Not implemented')

    def forward(self, X):
        raise Exception('Not implemented')

    def log_loss(self, X, y):
        a3 = self.forward(X)[-1]
        J = - y * np.log(a3).T
        J = np.sum(J)

        return J

    def function_prime(self, X, y):
        raise Exception('Not implemented')

    def modify_target(self, y):

        self.classes = []

        for x in set(y):
            self.classes.append(x)
        self.classes = np.array(self.classes)

        assert self.num_labels >= len(set(y))

        target = np.zeros((y.shape[0], self.num_labels))
        for i in range(y.shape[0]):
            target[i][self.classes == y[i]] = 1

        return target

    def fit(self,
            X_train, y_train,
            X_val=None, y_val=None,
            results_file=None,
            lr=0.1,
            eps=1e-6,
            rho=0.95,
            rho_up=1.2,
            information_freq=5,
            method='adadelta',
            reg=0.01,
            epoch=30,
            mini_batch_size=25,
            train_acc=False
            ):

        if results_file is None:
            results_file = u'results' + u'_' + unicode(self.vector_size) + u'_' + unicode(y_train.shape[0]) + u'_' + unicode(datetime.date.today())

        import Tee
        Tee.Tee(results_file)

        y = self.modify_target(y_train)

        weights = self.weights

        method = method.lower()
        if method == 'adadelta':
            descender = Adadelta(lr, reg=reg, rho=rho, eps=eps, **weights)
        elif method == 'adagrad':
            descender = Adagrad(lr, reg=reg, eps=eps, **weights)
        elif method == 'rmsprop':
            descender = RMSprop(lr, rho=rho, reg=reg, eps=eps, rho_up=rho_up, **weights)
        else:
            raise Exception('Unknow strategy for gradient descent: %s' % method)

        print 'Starting the gradient descent...', '\n'

        t0 = time.time()

        for j in range(epoch):

            print 'Epoch ', j, '\n'

            permuted_indexes = np.random.permutation(range(y.shape[0]))
            mini_batches = int(np.floor(y.shape[0] / mini_batch_size - 1))
            print 'Permutation computed'

            for i in range(y.shape[0] / mini_batch_size - 1):

                err = 0
                for k in range(mini_batch_size):
                    J, deltas = self.function_prime(X_train[permuted_indexes[i*mini_batch_size+k]][:, :].T, y[permuted_indexes[i*mini_batch_size+k]])
                    descender.acc(**deltas)
                    err += J

                # if i % 25 == 0:
                #     gradient_norm = descender.dWs.map_values(np.linalg.norm).fold(0, (lambda x, y: x + y))

                descender.end_batch(mini_batch_size)
                err /= mini_batch_size

                if i % 25 == 0:
                    t1 = time.time()
                    print i, 'mini-batches processed out of', mini_batches, '| cost:', err, '| time:', t1 - t0
                    # print 'gradient:', gradient_norm
                    t0 = t1

            print '100 % done!', '\n'

            if j % information_freq == 0:
                print 'Saving models...'
                self.save_model('%s_%i.pkl' % (results_file, j))
                print 'Saved!', '\n'

                print 'Results at epoch', j, '\n'
                print 'Validation Accuracy : ', self.accuracy(X_val, y_val)

                if train_acc:
                    print 'Training Accuracy : ', self.accuracy(X_train, y_train)

                print
                descender.information_display()

            if method == 'adadelta':
                descender.lr *= 0.95

        if (epoch - 1) % information_freq == 0:
            print 'Saving last model...'
            self.save_model('%s_end.pkl' % results_file)
            print 'Saved!', '\n'

            print 'Results at epoch', j, '\n'
            print 'Validation Accuracy : ', self.accuracy(X_val, y_val), '\n'
            print 'Training Accuracy:', self.accuracy(X_train, y_train), '\n'

    def accuracy(self, X, y):
        return np.mean([self.predict(X[i][:, :].T) for i in range(X.shape[0])] == y)

    def cost(self, X, y):
        err = 0
        n = X.shape[0]
        for i in range(n):
            err += self.log_loss(X[i][:, :].T, y[i])

        return err / n

    def information_display(self, X_train, y_train, X_val, y_val, y, epoch, descender):

        training_acc = self.accuracy(X_train, y_train)
        print 'Training Accuracy : ', training_acc, '\n'

        val_acc = self.accuracy(X_val, y_val)
        print 'Validation Accuracy : ', val_acc, '\n'

        cost = self.cost(X_train, y_train)
        print 'Cost at epoch : ', epoch, cost, '\n'

        descender.information_display()

    def conf_matrix(self, y_pred, y_gold):
        conf = np.zeros((self.num_labels, self.num_labels), dtype=int)
        for i in range(y_pred.size):
            conf[y_gold[i], y_pred[i]] += 1

        return conf

    def save_model(self, model_name):
        np.savez(model_name, **self.weights)

    def load_model(self, model_name):
        old_weights = self.weights
        new_weights = np.load(model_name)

        for k, weight in new_weights.iteritems():
            np.copyto(new_weights[k], old_weights[k])

    def predict(self, X):

        index = int(self.predict_proba(X).argmax(0))
        return self.classes[index]

    def predict_proba(self, X):

        return self.forward(X)[-1]

    def gradient_check(self, e=1e-8, eps=1e-5, n=5):
        from gradient_check import gradient_check as check

        X = np.random.rand(self.vector_size, n)
        y = np.zeros(self.num_labels)
        y[np.random.randint(0, self.num_labels)] = 1

        return check(self.function_prime, self.log_loss, self.weights, X, y, e=e, eps=eps)
