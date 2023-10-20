#!/usr/bin/env python
import unittest

import numpy as np

from epik.src.utils import (seq_to_one_hot, diploid_to_one_hot,
                            get_full_space_one_hot)


class UtilsTests(unittest.TestCase):
    def test_one_hot_encoding(self):
        X = np.array(['AA', 'AB', 'BA', 'BB'])
        x = seq_to_one_hot(X).numpy()
        onehot = np.array([[1, 0, 1, 0],
                           [1, 0, 0, 1],
                           [0, 1, 1, 0],
                           [0, 1, 0, 1.]])
        assert(np.allclose(x - onehot, 0))
    
    def test_get_full_one_hot(self):
        X = get_full_space_one_hot(seq_length=2, n_alleles=2)
        assert(np.all(X == [[1, 0, 1, 0],
                            [0, 1, 1, 0],
                            [1, 0, 0, 1],
                            [0, 1, 0, 1]]))
        
        X = get_full_space_one_hot(seq_length=2, n_alleles=3)
        assert(np.all(X == [[1, 0, 0, 1, 0, 0],
                            [0, 1, 0, 1, 0, 0],
                            [0, 0, 1, 1, 0, 0],
                            [1, 0, 0, 0, 1, 0],
                            [0, 1, 0, 0, 1, 0],
                            [0, 0, 1, 0, 1, 0],
                            [1, 0, 0, 0, 0, 1],
                            [0, 1, 0, 0, 0, 1],
                            [0, 0, 1, 0, 0, 1],]))
    
    def test_diploid_encoding(self):
        X = np.array(['00', '01', '11', '02', '22'])
        x = diploid_to_one_hot(X).numpy()
        assert(x.shape == (5, 2, 3))
        
        h0 = np.array([[1, 1], [1, 0], [0, 0], [1, 0], [0, 0]])
        ht = np.array([[0, 0], [0, 1], [1, 1], [0, 0], [0, 0]])
        h1 = np.array([[0, 0], [0, 0], [0, 0], [0, 1], [1, 1]])
        y = np.stack([h0, ht, h1], axis=2)
        assert(np.allclose(x, y))
    
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'UtilsTests']
    unittest.main()
