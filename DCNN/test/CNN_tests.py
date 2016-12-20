__author__ = 'jmm'

import numpy as np
import unittest

from DCNN.DCNN import CNN

class CNNTest(unittest.TestCase):

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

    def test_gradient_check(self):

        e = 1e-4
        X = np.random.rand(4,2)
        y = np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
        test = CNN(vector_size=4)
        CM1,B1,CM2,B2,FCL = test.initialize_weights()

        a0, z1, a1, kmax_indexes_layer1, z2, a2, kmax_indexes_layer2, z3, a3  = test._forward(X, CM1,B1,CM2,B2,FCL)
        d1fm, d2fm, d3 = test.function_prime(X, y, CM1,B1,CM2,B2,FCL)
        a2_packed = np.concatenate([a2[i].reshape(-1) for i in range(test.feature_maps_number_layer2)])
        a2_packed = np.concatenate(([1], a2_packed)).T
        kmax1 = int(np.max([test.kmax, np.ceil(0.5*X.shape[1])]))

        # Through FCL
        for i in range(test.num_labels):
            for j in range(test.kmax*test.feature_maps_number_layer2*test.folding_width + 1):

                FCL[i,j] += e
                Jpos = test.function(X,y,CM1,B1,CM2,B2,FCL)
                FCL[i,j] -= 2*e
                Jneg = test.function(X,y,CM1,B1,CM2,B2,FCL)

                J = (Jpos - Jneg)/(2*e)
                backprop = d3[i]*a2_packed[j]
                diff = J-backprop
                self.assertAlmostEquals(diff,0.0)
                FCL[i,j] += e

        #Through CL2
        for i in range(test.feature_maps_number_layer2):
            for j in range(test.feature_maps_number_layer1):
                for n in range(test.vector_size):
                    for p in range(test.second_kernel_size):

                        CM2[i][j][n,p] += e
                        Jpos = test.function(X,y,CM1,B1,CM2,B2,FCL)
                        CM2[i][j][n,p] -= 2*e
                        Jneg = test.function(X,y,CM1,B1,CM2,B2,FCL)

                        J = (Jpos - Jneg)/(2*e)
                        backprop =0
                        for v in range(kmax1+test.second_kernel_size-1):
                            if v-test.second_kernel_size+p-1 >= 0:
                                backprop += d2fm[i][n,v]*a1[j][n,v-test.second_kernel_size+p-1]
                        print J,backprop
                        self.assertAlmostEquals(J,backprop)
                        CM2[i][j][n,p] += e


