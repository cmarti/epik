#!/usr/bin/env python
import unittest

import numpy as np
import torch

from epik.src.utils import seq_to_one_hot, get_theta_to_log_lda_matrix


class UtilsTests(unittest.TestCase):
    def test_one_hot_encoding(self):
        X = np.array(['AA', 'AB', 'BA', 'BB'])
        x = seq_to_one_hot(X).numpy()
        onehot = np.array([[1, 0, 1, 0],
                           [1, 0, 0, 1],
                           [0, 1, 1, 0],
                           [0, 1, 0, 1.]])
        assert(np.allclose(x - onehot, 0))
    
    def test_theta_to_log_lda_matrix(self):
        m = get_theta_to_log_lda_matrix(l=7)
        print(m.numpy().astype(int))
        
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'UtilsTests']
    unittest.main()
