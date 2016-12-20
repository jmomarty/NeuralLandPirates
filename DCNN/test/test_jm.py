# -*- coding: utf-8 -*-
__author__ = 'jmm'

import numpy as np
from DCNN.DCNN import CNN
from DCNN.simple_convolve import convolve

def test_forward(self):

    X = np.zeros((2,2))
    X[0,0] = 1
    test = CNN(vector_size=2)
    CM1,B1,CM2,B2, FCL = test.initialize_weights()
    a0, z1, a1, k1, z2, a2, k2, z3, a3  = test._forward(X,CM1,B1,CM2,B2,FCL)
    print X, u'\n'
    print a0, u'\n'
    for i in range(test.feature_maps_number_layer1):
        print 'CM1',CM1[i], u'\n'
        print 'BM1',B1[i], u'\n'
        print 'z1',z1[i], u'\n'
        print 'a1',a1[i], u'\n'
        print 'kmaxindex',k1[i], u'\n'
        print u'\n\n'
    for i in range(test.feature_maps_number_layer2):
        for k in range(test.feature_maps_number_layer1):
            print 'CM2',CM2[i][k], u'\n'
        print 'B2',B2[i], u'\n'
        print 'z2',z2[i], u'\n'
        print 'a2',a2[i], u'\n'
        print 'kmaxindex',k2[i], u'\n'
        print u'\n\n'
    print 'z3',z3, u'\n'
    print 'a3',a3, u'\n'

def test_cost(self):

    X = np.random.rand(4,2)
    y = np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
    test = CNN(vector_size=4)
    CM1,B1,CM2,B2, FCL = test.initialize_weights()
    print test.function(X, y, CM1,B1,CM2,B2, FCL)

def test_function_prime(self):

    X = np.random.rand(4,2)
    y = np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
    test = CNN(vector_size=4)
    CM1,B1,CM2,B2, FCL = test.initialize_weights()
    d1fm, d2, d3 = test.function_prime(X, y,CM1,B1,CM2,B2, FCL)
    for i in range(test.feature_maps_number_layer1):
        print d1fm[i], u'\n'
    print d2, u'\n'
    print d3, u'\n'

def test_gradient_check(e):

    X = np.random.rand(4,2)
    y = np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
    test = CNN(vector_size=4)
    CM1,B1,CM2,B2,FCL = test.initialize_weights()

    dCM1, dB1, dCM2, dB2, dFCL = test.function_prime(X, y, CM1,B1,CM2,B2,FCL)

    # Through FCL
    for i in range(test.num_labels):
        for j in range(test.kmax*test.feature_maps_number_layer2*test.folding_width + 1):

            FCL[i,j] += e
            Jpos = test.function(X,y,CM1,B1,CM2,B2,FCL)
            FCL[i,j] -= 2*e
            Jneg = test.function(X,y,CM1,B1,CM2,B2,FCL)

            J = (Jpos - Jneg)/(2*e)
            backprop = dFCL[i,j]
            if (J-backprop)/(J+backprop)>e:
                print (J-backprop)/(J+backprop)
            FCL[i,j] += e

    # #Through CL2
    for i in range(test.feature_maps_number_layer2):
        for j in range(test.feature_maps_number_layer1):
            for n in range(test.vector_size):
                for p in range(test.second_kernel_size):

                    CM2[i][j][n,p] += e
                    Jpos = test.function(X,y,CM1,B1,CM2,B2,FCL)
                    CM2[i][j][n,p] -= 2*e
                    Jneg = test.function(X,y,CM1,B1,CM2,B2,FCL)

                    J = (Jpos - Jneg)/(2*e)
                    backprop = dCM2[i][j][n,p]
                    if (J-backprop)/(J+backprop)>e:
                        print (J-backprop)/(J+backprop)
                    CM2[i][j][n,p] += e

    for i in range(test.feature_maps_number_layer2):
        for j in range(test.folding_width):

            B2[i][j] += e
            Jpos = test.function(X,y,CM1,B1,CM2,B2,FCL)
            B2[i][j] -= 2*e
            Jneg = test.function(X,y,CM1,B1,CM2,B2,FCL)
            J = (Jpos - Jneg)/(2*e)
            backprop = dB2[i][j]
            if (J-backprop)/(J+backprop)>e:
                print (J-backprop)/(J+backprop)
            B2[i][j] += e

    # Through CL1
    for i in range(test.feature_maps_number_layer1):
        for j in range(test.vector_size):

            B1[i][j] += e
            Jpos = test.function(X,y,CM1,B1,CM2,B2,FCL)
            B1[i][j] -= 2*e
            Jneg = test.function(X,y,CM1,B1,CM2,B2,FCL)
            J = (Jpos - Jneg)/(2*e)
            backprop = dB1[i][j]
            if (J-backprop)/(J+backprop)>e:
                print (J-backprop)/(J+backprop)
            B1[i][j] += e

    for i in range(test.feature_maps_number_layer1):
        for u in range(test.vector_size):
            for v in range(test.first_kernel_size):

                CM1[i][u,v] += e
                Jpos = test.function(X,y,CM1,B1,CM2,B2,FCL)
                CM1[i][u,v] -= 2*e
                Jneg = test.function(X,y,CM1,B1,CM2,B2,FCL)
                J = (Jpos - Jneg)/(2*e)
                backprop = dCM1[i][u,v]
                if (J-backprop)/(J+backprop)>e:
                    print (J-backprop)/(J+backprop)
                CM1[i][u,v] += e

def test_fit():

    test_data_file = np.load("../training_arrays-1000.npz")
    X, y, z= test_data_file['X_train'],test_data_file['y_train'], test_data_file['z_train']
    test = CNN(vector_size=100)
    test.fit(X,z)

def test_convolve():

    x = np.array([1,2,3])
    m = np.array([1,2])
    print np.convolve(x,m)

def test_ReLU():

    A = np.random.rand(4,4)-0.5
    test = CNN(vector_size=100)
    print A,'\n',test.ReLU(A),'\n', test.ReLU_prime(A)

def test_new_pack_unpack():

    test = CNN(vector_size=4)
    A, B, C, D, E, F = test.unpack_weights(test.weights)
    print A ==  test.M1
    print '\n\n'
    print B , '\n\n', test.CM1
    print '\n\n'
    print C , '\n\n', test.B1
    print '\n\n'
    print D, '\n\n',  test.CM2
    print '\n\n'
    print E, '\n\n', test.B2
    print '\n\n'
    print F == test.FCL
    print '\n\n'

if __name__ == "__main__":

    test_gradient_check(1e-6)

