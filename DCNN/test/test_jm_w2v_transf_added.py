# -*- coding: utf-8 -*-
__author__ = 'jmm'

import numpy as np
from DCNN.DCNN_with_W2V_transf import CNN
from DCNN.simple_convolve import convolve

def test_forward():

    X = np.zeros((2,2))
    X[0,0] = 1
    test = CNN(vector_size=2)
    W1, _B1, CM1,B1,CM2,B2, FCL = test.initialize_weights()
    z0, a0, z1, a1, z2, a2, z3, a3  = test._forward(X,W1, _B1,CM1,B1,CM2,B2,FCL)
    print X, u'\n'
    print z0, u'\n'
    print a0, u'\n'
    for i in range(test.feature_maps_number_layer1):
        print 'CM1',CM1[i], u'\n'
        print 'BM1',B1[i], u'\n'
        print 'z1',z1[i], u'\n'
        print 'a1',a1[i], u'\n'
        print u'\n\n'
    for i in range(test.feature_maps_number_layer2):
        for k in range(test.feature_maps_number_layer1):
            print 'CM2',CM2[i][k], u'\n'
        print 'B2',B2[i], u'\n'
        print 'z2',z2[i], u'\n'
        print 'a2',a2[i], u'\n'
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
    W1,_B1, CM1,B1,CM2,B2,FCL = test.initialize_weights()

    dW1, d_B1, dCM1, dB1, dCM2, dB2, dFCL = test.function_prime(X, y,W1, _B1, CM1,B1,CM2,B2,FCL)

    # Through FCL
    for i in range(test.num_labels):
        for j in range(test.kmax*test.feature_maps_number_layer2*test.folding_width + 1):

            FCL[i,j] += e
            Jpos = test.function(X,y,W1, _B1,CM1,B1,CM2,B2,FCL)
            FCL[i,j] -= 2*e
            Jneg = test.function(X,y,W1, _B1,CM1,B1,CM2,B2,FCL)

            J = (Jpos - Jneg)/(2*e)
            backprop = dFCL[i,j]
            if (J-backprop)/(J+backprop)>e:
                print (J-backprop)/(J+backprop)
                print J, backprop
            FCL[i,j] += e

    # #Through CL2
    for i in range(test.feature_maps_number_layer2):
        for j in range(test.feature_maps_number_layer1):
            for n in range(test.vector_size):
                for p in range(test.second_kernel_size):

                    CM2[i][j][n,p] += e
                    Jpos = test.function(X,y,W1, _B1,CM1,B1,CM2,B2,FCL)
                    CM2[i][j][n,p] -= 2*e
                    Jneg = test.function(X,y,W1, _B1,CM1,B1,CM2,B2,FCL)

                    J = (Jpos - Jneg)/(2*e)
                    backprop = dCM2[i][j][n,p]
                    if (J-backprop)/(J+backprop)>e:
                        print (J-backprop)/(J+backprop)
                        print J, backprop
                    CM2[i][j][n,p] += e

    for i in range(test.feature_maps_number_layer2):
        for j in range(test.folding_width):

            B2[i][j] += e
            Jpos = test.function(X,y,W1, _B1,CM1,B1,CM2,B2,FCL)
            B2[i][j] -= 2*e
            Jneg = test.function(X,y,W1, _B1,CM1,B1,CM2,B2,FCL)
            J = (Jpos - Jneg)/(2*e)
            backprop = dB2[i][j]
            if (J-backprop)/(J+backprop)>e:
                print (J-backprop)/(J+backprop)
                print J, backprop
            B2[i][j] += e

    # Through CL1
    for i in range(test.feature_maps_number_layer1):
        for j in range(test.vector_size):

            B1[i][j] += e
            Jpos = test.function(X,y,W1, _B1,CM1,B1,CM2,B2,FCL)
            B1[i][j] -= 2*e
            Jneg = test.function(X,y,W1, _B1,CM1,B1,CM2,B2,FCL)
            J = (Jpos - Jneg)/(2*e)
            backprop = dB1[i][j]
            if (J-backprop)/(J+backprop)>e:
                print (J-backprop)/(J+backprop)
                print J, backprop
            B1[i][j] += e

    for i in range(test.feature_maps_number_layer1):
        for u in range(test.vector_size):
            for v in range(test.first_kernel_size):

                CM1[i][u,v] += e
                Jpos = test.function(X,y,W1, _B1,CM1,B1,CM2,B2,FCL)
                CM1[i][u,v] -= 2*e
                Jneg = test.function(X,y,W1, _B1,CM1,B1,CM2,B2,FCL)
                J = (Jpos - Jneg)/(2*e)
                backprop = dCM1[i][u,v]
                if (J-backprop)/(J+backprop)>e:
                    print (J-backprop)/(J+backprop)
                    print J, backprop
                CM1[i][u,v] += e

    # Through W2VTransf
    for i in range(test.vector_size):
        for j in range(X.shape[1]):

            W1[i,j] += e
            Jpos = test.function(X,y,W1, _B1,CM1,B1,CM2,B2,FCL)
            W1[i,j] -= 2*e
            Jneg = test.function(X,y,W1, _B1,CM1,B1,CM2,B2,FCL)

            J = (Jpos - Jneg)/(2*e)
            backprop = dW1[i,j]
            if (J-backprop)/(J+backprop)>e:
                print (J-backprop)/(J+backprop)
            W1[i,j] += e

    # Through W2VTransf
    for i in range(test.vector_size):

        _B1[i] += e
        Jpos = test.function(X,y,W1, _B1,CM1,B1,CM2,B2,FCL)
        _B1[i] -= 2*e
        Jneg = test.function(X,y,W1, _B1,CM1,B1,CM2,B2,FCL)

        J = (Jpos - Jneg)/(2*e)
        backprop = d_B1[i]
        if (J-backprop)/(J+backprop)>e:
            print (J-backprop)/(J+backprop)
        _B1[i] += e



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
    A, B, C, D, E, F, G = test.unpack_weights(test.weights)
    print A ==  test.w2v_transf_matrix_1
    print '\n\n'
    print B == test.w2v_transf_bias_1
    print '\n\n'
    for x in C.keys():
        print C[x] == test.CM1[x]
    print '\n\n'
    for x in D.keys():
        print D[x]== test.B1[x]
    print '\n\n'
    for x in E.keys():
        for y in E[x].keys():
            print E[x][y]==test.CM2[x][y]
    print '\n\n'
    for x in F.keys():
        print F[x]== test.B2[x]
    print '\n\n'
    print G == test.FCL
    print '\n\n'

if __name__ == "__main__":

    test_gradient_check(1e-6)

