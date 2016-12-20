# -*- coding: utf-8 -*-
import numpy as np

from convolve import convolution, correlate, kernel_gradient, pool, unpool
from base_dcnn import AbstractCNN
from util import rand_init, tanh, tanh_prime, softmax, dotBias


class CNN(AbstractCNN):

    def __init__(self,
                 first_kernel_size=6,
                 fmap1=3,
                 second_kernel_size=5,
                 fmap2=4,
                 kmax=3,
                 num_labels=15,
                 vector_size=30,
                 num_Ws=4,
                 ):

        super(CNN, self).__init__(kmax, num_labels, vector_size)

        # Activation Function & its Derivative
        self.activation_func = tanh
        self.activation_func_prime = tanh_prime

        self.W_func = tanh
        self.W_func_prime = tanh_prime
        # self.W_func = self.ReLU
        # self.W_func_prime = self.ReLU_prime

        # Useful Size Variables
        self.vector_size = vector_size
        self.first_kernel_size = first_kernel_size
        self.second_kernel_size = second_kernel_size
        self.kmax = kmax
        self.kmax1 = self.kmax
        self.fmap1 = fmap1
        self.fmap2 = fmap2
        self.num_labels = num_labels

        assert self.vector_size % 2 == 0
        self.folding_width = self.vector_size / 2

        # W2V transformation layer:
        self.num_Ws = num_Ws

        # Weights
        self.Ws, self.Ws_bias, self.CM1, self.B1, self.CM2, self.B2, self.FCL = self.initialize_weights()
        self.weights = {'Ws': self.Ws, 'Ws_bias': self.Ws_bias,
                        'CM1': self.CM1, 'B1': self.B1,
                        'CM2': self.CM2, 'B2': self.B2,
                        'FCL': self.FCL
                        }

        # K-Max Pooling Indexes Matrices
        self._off_layer1 = np.arange(self.fmap1 * self.vector_size).reshape(self.fmap1, self.vector_size, 1)
        self._off_layer2 = np.arange(self.fmap2 * self.folding_width).reshape(self.fmap2, self.folding_width, 1)

    def unwrap_weights(self, weights=None):
        if weights is None:
            weights = self.weights
        w = weights
        return w['Ws'], w['Ws_bias'], w['CM1'], w['B1'], w['CM2'], w['B2'], w['FCL']

    def initialize_weights(self):

        # Initialize the neurons

        # Layers to transform word2vecs
        Ws = np.zeros((self.num_Ws, self.vector_size, self.vector_size))
        Ws_bias = np.zeros((self.num_Ws, self.vector_size, 1))
        for i in range(self.num_Ws):
            Ws[i] = rand_init(self.vector_size, self.vector_size)

        # Layer 1 : A different kernel/bias for each feature map at layer 1
        CM1 = np.zeros((self.fmap1, 1, self.vector_size, self.first_kernel_size))
        B1 = np.zeros((self.fmap1, self.vector_size, 1))

        # Layer 2 : A different kernel/bias for each feature map at layer 2 and for each feature map at layer 1
        CM2 = np.zeros((self.fmap2, self.fmap1, self.vector_size, self.second_kernel_size))
        B2 = np.zeros((self.fmap2, self.folding_width, 1))

        # Layer 1 : intializing the feature maps weights for layer 1 (biases + kernels)
        for i in range(self.fmap1):
            CM1[i] = rand_init(self.vector_size, self.first_kernel_size)

        # Layer 2
        for i in range(self.fmap2):
            for j in range(self.fmap1):
                CM2[i, j] = rand_init(self.vector_size, self.second_kernel_size)

        #Layer 3 (Fully Connected Layer)
        FCL = rand_init(self.num_labels, self.kmax*self.fmap2*self.folding_width + 1)
        FCL[:, -1] = 0

        return Ws, Ws_bias, CM1, B1, CM2, B2, FCL

    def forward(self, X):

        """
        Forward Propagation : self.num_Ws fully word embedding layers, then 2 convolutions layers (with kmax pooling)
                              then 1 folding layer, then 1 fully connected layer
        :param X: Sentence-Matrix to process, 3d Float Numpy Array
        :return:
        a0 : 3d Float Numpy Array
        z1 : 3d Float Numpy Array
        a1 : 3d Float Numpy Array
        z2_pool : 3d Float Numpy Array
        a2 : 4d Float Numpy Array
        z3 : 2d Float Numpy Array
        a3 : 2d Float Numpy Array
        """

        a0 = np.zeros((self.num_Ws + 1, self.vector_size, X.shape[1]))
        a0[0] = X

        # Transform each vector
        for i in range(self.num_Ws):
            z0 = np.dot(self.Ws[i], a0[i]) + self.Ws_bias[i]
            a0[i + 1] = self.W_func(z0)

        # Wide Convolutional Layer 1

        # Kmax-Pooling + Translation
        self.kmax1 = int(np.max([self.kmax, np.ceil(0.5*X.shape[1])]))

        # Convolution
        c1 = convolution(self.CM1, a0[-1].reshape(1, self.vector_size, X.shape[1]))

        # Kmax-Pooling
        z1, indexes1 = pool(c1, self.kmax1, self._off_layer1)

        # Translation
        z1 += self.B1

        # Non-linearity
        a1 = self.activation_func(z1)

        # Wide Convolutional Layer 2 & Folding

        # Convolution
        c2 = convolution(self.CM2, a1)

        # Folding
        z2 = c2[:, :self.folding_width, :] + c2[:, self.folding_width:, :]

        # K-max Pooling
        z2_pool, indexes2 = pool(z2, self.kmax, self._off_layer2)

        # Translation
        z2_pool += self.B2

        # Non-linearity
        a2 = self.activation_func(z2_pool).reshape(-1)

        # Fully connected layer
        z3 = dotBias(self.FCL, a2)
        a3 = softmax(z3)

        return a0, indexes1, a1, indexes2, a2, a3

    def function_prime(self, X, y):

        """
        Backward Propagation
        :param X: Sentence-Matrix to process, 3d Float Numpy Array
        :param y: Target-Vector, 2d Integer Numpy Array
        :return:
        J : Cost, Float
        dWs : 3d Float Numpy Array
        dWs_b : 2d Float Numpy Array
        dCM1 : 3d Float Numpy Array
        dB1 : 2d Float Numpy Array
        dCM2 : 4d Float Numpy Array
        dB2 : 3d Float Numpy Array
        dFCL : 2d Float Numpy Arrays
        """

        a0, indexes1, a1, indexes2, a2, a3 = self.forward(X)

        J = np.sum(- y * np.log(a3).T)

        length2 = self.second_kernel_size + self.kmax1 - 1
        length1 = X.shape[1] + self.first_kernel_size - 1

        # Backpropagation

        # Delta at the last layer
        delta_a3 = a3 - y

        # FCL gradient:
        a2_packed = np.concatenate((a2, [1]))
        dFCL = np.outer(delta_a3, a2_packed)

        # Delta at the FC layer
        FCLb = self.FCL[:, :-1]

        delta_a2 = np.dot(FCLb.T, delta_a3)

        delta_z2_pool = (delta_a2 * self.activation_func_prime(a2)).reshape(self.fmap2, self.folding_width, self.kmax)

        # B2 gradient:
        dB2 = np.sum(delta_z2_pool, axis=2).reshape(self.fmap2, self.folding_width, 1)

        # Through 2nd Layer
        # Through k-max pooling layer:
        delta_z2 = unpool(delta_z2_pool, indexes2, length2)

        # Through folding layer
        delta_c2 = np.zeros((self.fmap2, self.vector_size, length2))
        delta_c2[:, :self.folding_width, :] = delta_z2
        delta_c2[:, self.folding_width:, :] = delta_z2

        # CM2 gradient:
        dCM2 = kernel_gradient(delta_c2, a1)

        # Through 1st Layer

        # Through CM2
        delta_z1 = correlate(self.CM2, delta_c2) * self.activation_func_prime(a1)
        dB1 = np.sum(delta_z1, axis=2).reshape(self.fmap1, self.vector_size, 1)

        # Through K-Max Pooling Layer:
        delta_c1 = unpool(delta_z1, indexes1, length1)

        # CM1 gradient:
        dCM1 = kernel_gradient(delta_c1, a0[-1].reshape(1, self.vector_size, -1))

        # B1 gradient:
        n = self.num_Ws
        delta_a0 = np.zeros((n+1, self.vector_size, X.shape[1]))

        # Through CM1
        delta_a0[n] = correlate(self.CM1, delta_c1) * self.W_func_prime(a0[n])
        delta_a0[n] = delta_a0[n].reshape(self.vector_size, X.shape[1])

        # Through W2V Transformation Layers:
        dWs = np.zeros((n, self.vector_size, self.vector_size))
        dWs_b = np.zeros((n, self.vector_size, 1))

        for i in range(n)[::-1]:
            delta_a0[i] = np.dot(self.Ws[i].T, delta_a0[i + 1])
            delta_a0[i] *= self.W_func_prime(a0[i])
            dWs[i] = np.dot(delta_a0[i + 1], a0[i].T)
            dWs_b[i] = np.sum(delta_a0[i + 1], axis=1).reshape((-1, 1))

        return J, {'Ws': dWs, 'Ws_bias': dWs_b,
                   'CM1': dCM1, 'B1': dB1,
                   'CM2': dCM2, 'B2': dB2, 'FCL': dFCL
                   }
