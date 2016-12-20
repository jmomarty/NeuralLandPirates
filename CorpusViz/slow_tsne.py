#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python 2.5.1, and it requires a working
# installation of NumPy. The implementation comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
# The example can be run by executing: ipython tsne.py -pylab
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import numpy as np
import pylab as plt

import DCNN.gradient_step as gradient


def Hbeta(D=np.array([]), beta=1.0):
    """Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""

    # Compute P-row and corresponding perplexity

    # original version, unstable for big values of D
    # P = np.exp(- beta * D)
    # sumP = sum(P)
    # H = np.log(sumP) + beta * np.sum(D * P) / sumP
    # P = P / sumP

    beta_D = beta * D
    beta_D_min = beta_D.min()
    P_shifted = np.exp(- (beta_D - beta_D_min))
    sum_P_shifted = P_shifted.sum()

    P_shift = np.exp(- beta_D_min)

    H = - beta_D_min + np.log(sum_P_shifted) + np.sum( beta_D * P_shifted) / sum_P_shifted
    P = P_shifted / sum_P_shifted

    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0, sym=True, normalize=True):
    """Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""

    # Initialize some variables
    print "Computing pairwise distances..."
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print "Computing P-values for point ", i, " of ", n, "..."

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i]
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i]
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries = tries + 1

        # Set the final row of P
        if np.isnan(thisP.sum()):
            print 'error on', i
            print '\tbeta:', beta[i]
            print '\Di:', Di
        else:
            P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    sigma = np.mean(np.sqrt(1 / beta))
    print "Mean value of sigma: ", sigma

    # Return final P-matrix
    if sym:
        P += P.T
    if normalize:
        P /= np.sum(P)
    return P


def pca(X=np.array([]), no_dims=50, return_pi=False):
    """Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""

    print "Using PCA to reduce X: %s to only %s dimensions" % (X.shape, no_dims)
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))

    pi = M[:, 0:no_dims]
    Y = np.dot(X, pi)

    if return_pi:
        return Y, pi
    else:
        return Y


def y2q(Y):
    n, d = Y.shape
    # Compute pairwise affinities
    sum_Y = np.sum(np.square(Y), 1)
    num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y))
    num[range(n), range(n)] = 0
    Q = num / np.sum(num)
    Q = np.maximum(Q, 1e-12)
    return num, Q


def deltaY(P, Y, num, Q, out=None):
    n, no_dims = Y.shape

    dY = np.zeros(Y.shape) if out is None else out

    PQ = P - Q
    for i in range(n):
        dY[i, :] = np.mean(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

    return dY


def kl_loss(P, Q):
    return np.sum(P * np.log(P / Q))


def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0, max_iter=1000, reg=0, eta=500000, interactive=False, sgd='momentum'):
    """Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
    The syntaxis of the function is Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""

    # Check inputs
    # if X.dtype != "float64":
    #     print "Error: array X should have type float64."
    #     return -1
    #if no_dims.__class__ != "<type 'int'>":            # doesn't work yet!
    #    print "Error: number of dimensions should be an integer."
    #    return -1

    # Initialize variables
    if X.shape[1] > initial_dims:
        X = pca(X, initial_dims)
    (n, d) = X.shape

    initial_momentum = 0.5
    final_momentum = 0.8

    Y = np.random.randn(n, no_dims)

    if interactive:
        plt.ion()
        plt.scatter(Y[:, 0], Y[:, 1], 20)
        plt.pause(0.0001)

    dY = np.zeros(Y.shape)

    if sgd == 'adam':
        descender = gradient.Adam(eta, Y=Y, reg=reg, beta1=0.8, beta2=0.95)
    elif sgd == 'adadelta':
        descender = gradient.Adadelta(eta * 10, reg=0.002, Y=Y)
    elif sgd == 'momentum':
        descender = gradient.Momentum(eta, Y=Y, reg=reg, rho_init=initial_momentum, rho=final_momentum)
    else:
        raise Exception('invalid sgd algorithm')

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P * 4         # early exaggeration
    P = np.maximum(P, 1e-12)
    print 'P:', 'min', P.min(), 'max', P.max(), 'mean', P.mean()

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        num, Q = y2q(Y)

        # Compute gradient
        dY = deltaY(P, Y, num, Q, out=dY)

        # Perform the update
        descender.acc(Y=dY)
        descender.end_batch()
        Y -= np.mean(Y, 0)


        # display state on screen
        if interactive:
            plt.clf()
            plt.scatter(Y[:, 0], Y[:, 1], 20)
            plt.pause(0.0000000001)

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = kl_loss(P, Q) + 0.5 * reg / Y.size * np.sum(Y * Y)
            print "Iteration ", (iter + 1), ": error is ", C

            for k, lr in descender.learning_rates():
                print k, 'lr:', 'min', lr.min(), 'max', lr.max(), 'mean', lr.mean()

        # Stop lying about P-values
        if iter == 100:
            print 'Stop lying about P-values'
            P = P / 4

    if interactive:
        plt.show()

    # Return solution
    return Y


if __name__ == "__main__":
    print "Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset."
    print "Running example on 2,500 MNIST digits..."
    X = np.loadtxt("mnist2500_X.txt")
    labels = np.loadtxt("mnist2500_labels.txt")
    Y = tsne(X, 2, 50, 20.0, interactive=True)
    plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
