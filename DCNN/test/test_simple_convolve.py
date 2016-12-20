# -*- coding: utf8 -*-
import unittest

import numpy as np

from DCNN.simple_convolve import convolve, convolve_reverse, convolve1D


class ConvolveTest(unittest.TestCase):

    def test_convolve_produces_expected(self):
        x = np.array([1, 2, 1, 4, 1])
        k = np.array([-1, 2, -2])

        y = convolve(x, k, wide=0)
        self.assertEqual(list(y), [-2, 1, -8, 5, -2])

        y = convolve(x, k, wide=1)
        self.assertEqual(list(y), [-2, -2, 1, -8, 5, -2, -1])

        y = convolve(x, k, wide=-1)
        self.assertEqual(list(y), [1, -8, 5])

    def test_convolve_reverse_produces_expected(self):
        k = np.array([-1, 2, -2])

        y = np.array([1, -1, 2, 3, 1])
        x = convolve_reverse(y, k, wide=0)
        self.assertEqual(list(x), [3, -6, 3, 1, -4])

        y = np.array([0, 1, -1, 2, 3, 1, 4])
        x = convolve_reverse(y, k, wide=1)
        self.assertEqual(list(x), [3, -6, 3, 1, -8])

        y = np.array([-1, 2, 3])
        x = convolve_reverse(y, k, wide=-1)
        self.assertEqual(list(x), [1, -4, 3, 2, -6])

    def test_convolve1D_produces_expected(self):
        x = np.array([[1, 2, 1, 4, 1], [1, 0, 1, 3, -1]])
        k = np.array([[-1, 2, -2], [1, 0, -1]])

        y = convolve1D(x, k, wide=0, axis=1)
        self.assertEqual(list(y[0, :]), [-2, 1, -8, 5, -2])
        self.assertEqual(list(y[1, :]), [-0, 0, -3, 2,  3])

        y = convolve1D(x, k, wide=1, axis=1)
        self.assertEqual(list(y[0, :]), [-2, -2, 1, -8, 5, -2, -1])
        self.assertEqual(list(y[1, :]), [-1,  0, 0, -3, 2,  3, -1])

        y = convolve1D(x, k, wide=-1, axis=1)
        self.assertEqual(list(y[0, :]), [1, -8, 5])
        self.assertEqual(list(y[1, :]), [0, -3, 2])
