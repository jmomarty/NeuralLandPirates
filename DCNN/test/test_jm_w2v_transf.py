# -*- coding: utf-8 -*-
__author__ = 'jmm'

import numpy as np
from DCNN.DCNN_with_W2V_transf import CNN

def test_forward():

    X = np.zeros((2,2))
    X[0,0] = 1
    test = CNN(vector_size=2)
    W1, dict = test.initialize_weights()
    _Ws,_Bs, CM1,B1,CM2,B2, FCL = dict['Ws'], dict['Ws_bias'], dict['CM1'], dict['B1'], dict['CM2'], dict['B2'], dict['FCL']
    z0, a0, z1, a1, z2, a2, z3, a3  = test.forward(X)
    print X, u'\n'
    print z0, u'\n'
    print a0, u'\n'
    for i in range(test.fmap1):
        print 'CM1',CM1[i], u'\n'
        print 'BM1',B1[i], u'\n'
        print 'z1',z1[i], u'\n'
        print 'a1',a1[i], u'\n'
        print u'\n\n'
    for i in range(test.fmap2):
        for k in range(test.fmap1):
            print 'CM2',CM2[i][k], u'\n'
        print 'B2',B2[i], u'\n'
        print 'z2',z2[i], u'\n'
        print 'a2',a2[i], u'\n'
        print u'\n\n'
    print 'z3',z3, u'\n'
    print 'a3',a3, u'\n'

def test_function_prime(self):

    X = np.random.rand(4,2)
    y = np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
    test = CNN(vector_size=4)
    CM1,B1,CM2,B2, FCL = test.initialize_weights()
    d1fm, d2, d3 = test.function_prime(X, y)
    for i in range(test.fmap1):
        print d1fm[i], u'\n'
    print d2, u'\n'
    print d3, u'\n'

def test_gradient_check(e):

    X = np.random.rand(4,2)
    y = np.random.permutation(np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    test = CNN(vector_size=4, num_Ws=4)
    return test.gradient_check()

    # W,B,CM1,B1,CM2,B2,FCL = test.initialize_weights()
    #
    # J, dict = test.function_prime(X, y)
    # dW,dB, dCM1,dB1,dCM2,dB2, dFCL = dict['Ws'], dict['Ws_bias'], dict['CM1'], dict['B1'], dict['CM2'], dict['B2'], dict['FCL']
    #
    # # Through FCL
    # print "\nThrough FCL Matrix \n"
    # for i in range(test.num_labels):
    #     for j in range(test.kmax*test.fmap2*test.folding_width + 1):
    #
    #         FCL[i,j] += e
    #         Jpos = test.log_loss(X,y)
    #         FCL[i,j] -= 2*e
    #         Jneg = test.log_loss(X,y)
    #
    #         J = (Jpos - Jneg)/(2*e)
    #         backprop = dFCL[i,j]
    #         if (J-backprop)>e*10:
    #             print (J-backprop)
    #             print J, backprop
    #         FCL[i,j] += e
    #
    # # Through CL2
    # print "\nThrough CL2 Matrices \n"
    # for i in range(test.fmap2):
    #     for j in range(test.fmap1):
    #         for n in range(test.vector_size):
    #             for p in range(test.second_kernel_size):
    #
    #                 CM2[i,j,n,p] += e
    #                 Jpos = test.log_loss(X,y)
    #                 CM2[i,j,n,p] -= 2*e
    #                 Jneg = test.log_loss(X,y)
    #
    #                 J = (Jpos - Jneg)/(2*e)
    #                 backprop = dCM2[i,j,n,p]
    #                 if (J-backprop)>e*10:
    #                     print (J-backprop)
    #                     print J, backprop
    #                 CM2[i,j,n,p] += e
    #
    # print "\nThrough CL2 Bias \n"
    # for i in range(test.fmap2):
    #     for j in range(test.folding_width):
    #
    #         B2[i,j] += e
    #         Jpos = test.log_loss(X,y)
    #         B2[i,j] -= 2*e
    #         Jneg = test.log_loss(X,y)
    #
    #         J = (Jpos - Jneg)/(2*e)
    #         backprop = dB2[i,j]
    #         if (J-backprop)>e:
    #             print (J-backprop)
    #             print J, backprop
    #         B2[i,j] += e
    #
    # # Through CL1
    # print "\nThrough CL1 Bias \n"
    # for i in range(test.fmap1):
    #     for j in range(test.vector_size):
    #
    #         B1[i,j,0] += e
    #         Jpos = test.log_loss(X,y)
    #         B1[i,j,0] -= e
    #         Jneg = test.log_loss(X,y)
    #
    #         J = (Jpos - Jneg)/e
    #         backprop = dB1[i,j,0]
    #         if (J-backprop)>e:
    #             print (J-backprop)
    #             print J, backprop
    #         B1[i,j,0] += e
    #
    # print "\nThrough CL1 Matrices \n"
    # for i in range(test.fmap1):
    #     for u in range(test.vector_size):
    #         for v in range(test.first_kernel_size):
    #
    #             CM1[i,u,v] += e
    #             Jpos = test.log_loss(X,y)
    #             CM1[i,u,v] -= 2*e
    #             Jneg = test.log_loss(X,y)
    #
    #             J = (Jpos - Jneg)/(2*e)
    #             backprop = dCM1[i,u,v]
    #             if (J-backprop)>e:
    #                 print (J-backprop)
    #                 print J, backprop
    #             CM1[i,u,v] += e
    #
    # # Through W2VTransf
    # print "\nW2VTransf Matrices\n"
    # for k in range(test.num_Ws):
    #     for i in range(test.vector_size):
    #         for j in range(test.vector_size):
    #
    #             W[k,i,j] += e
    #             Jpos = test.log_loss(X,y)
    #             W[k,i,j] -= 2*e
    #             Jneg = test.log_loss(X,y)
    #
    #             J = (Jpos - Jneg)/(2*e)
    #             backprop = dW[k,i,j]
    #             if (J-backprop)/(J+backprop)>e:
    #                 print (J-backprop)/(J+backprop)
    #             W[k,i,j] += e
    #
    # #Through W2VTransf
    # print "\nW2VTransf Bias \n"
    # for k in range(test.num_Ws):
    #     for i in range(test.vector_size):
    #
    #         B[k,i,0] += e
    #         Jpos = test.log_loss(X,y)
    #         B[k,i,0] -= e
    #         Jneg = test.log_loss(X,y)
    #
    #         J = (Jpos - Jneg)/e
    #         backprop = dB[k,i,0]
    #         if (J-backprop)>e:
    #             print (J-backprop)
    #             print J, backprop
    #         B[k,i,0] += e

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

