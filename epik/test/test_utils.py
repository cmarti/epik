#!/usr/bin/env python
import unittest

import numpy as np

from epik.src.utils import seq_to_one_hot


class UtilsTests(unittest.TestCase):
    def test_one_hot_encoding(self):
        X = np.array(['AA', 'AB', 'BA', 'BB'])
        x = seq_to_one_hot(X).numpy()
        onehot = np.array([[1, 0, 1, 0],
                           [1, 0, 0, 1],
                           [0, 1, 1, 0],
                           [0, 1, 0, 1.]])
        assert(np.allclose(x - onehot, 0))
    
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'UtilsTests']
    unittest.main()
