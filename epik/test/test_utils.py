#!/usr/bin/env python
import unittest

import numpy as np
from timeit import timeit

from epik.src.utils import (seq_to_one_hot, diploid_to_one_hot,
                            get_full_space_one_hot, one_hot_to_seq, 
                            encode_seq, get_one_hot_subseq_key,
                            get_binary_subseq_key, encode_seqs,
                            seq_to_binary, calc_decay_rates)


class UtilsTests(unittest.TestCase):
    def test_encode_seq(self):
        # Binary encoding
        seq = 'ABBA'
        subseq_key = get_binary_subseq_key(alphabet='AB')
        assert(len(subseq_key) == 2)
        
        encoding = encode_seq(seq, subseq_key)
        assert(encoding == [1, -1, -1, 1])
        assert(len(subseq_key) == 5)
        
        # Odd length
        seq = 'ABBAB'
        encoding = encode_seq(seq, subseq_key)
        assert(encoding == [1, -1, -1, 1, -1])
        assert(len(subseq_key) > 5)
        
        # Fail with missing characters
        seq = 'ABBAC'
        try:
            encoding = encode_seq(seq, subseq_key)
            self.fail()
        except ValueError:
            pass
        
        # One hot encoding
        seq = 'ACGT'
        subseq_key = get_one_hot_subseq_key(alphabet=seq)
        assert(len(subseq_key) == 4)
        
        encoding = encode_seq(seq, subseq_key)
        assert(encoding == [1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 1, 0,
                            0, 0, 0, 1])
        assert(len(subseq_key) == 7)
        
        encoding = encode_seq(seq[:3], subseq_key)
        assert(encoding == [1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 1, 0])
    
    def test_encode_seqs(self):
        seqs = np.array(['AA', 'AB', 'BA', 'BB'])
        alphabet = 'AB'
        
        onehot = np.array([[1, 0, 1, 0],
                           [1, 0, 0, 1],
                           [0, 1, 1, 0],
                           [0, 1, 0, 1.]])
        X = encode_seqs(seqs, alphabet, encoding_type='one_hot')
        assert(np.allclose(X, onehot))
        
        binary = np.array([[ 1,  1],
                           [ 1, -1],
                           [-1,  1],
                           [-1, -1]])
        X = encode_seqs(seqs, alphabet, encoding_type='binary')
        assert(np.allclose(X, binary))
        
    def test_time_encoding(self):
        alphabet = [a for a in 'ACGT']
        l = 10
        n = 100
        seqs = np.array([''.join(x) for x in np.random.choice(alphabet, size=(n, l))])
    
        print(timeit(lambda: encode_seqs(seqs, alphabet, max_n=4), number=10))
        print(timeit(lambda: seq_to_one_hot(seqs, alphabet), number=10))
        
    def test_one_hot_encoding(self):
        X = np.array(['AA', 'AB', 'BA', 'BB'])
        x = seq_to_one_hot(X).numpy()
        onehot = np.array([[1, 0, 1, 0],
                           [1, 0, 0, 1],
                           [0, 1, 1, 0],
                           [0, 1, 0, 1.]])
        assert(np.allclose(x - onehot, 0))
        
    def test_binary_encoding(self):
        X = np.array(['AA', 'AB', 'BA', 'BB'])
        x = seq_to_binary(X, ref='A').numpy()
        onehot = np.array([[ 1, 1.],
                           [ 1,-1],
                           [-1, 1],
                           [-1,-1]])
        assert(np.allclose(x, onehot))
    
    def test_one_hot_encoding_to_seq(self):
        x = np.array([[1, 0, 1, 0],
                      [1, 0, 0, 1],
                      [0, 1, 1, 0],
                      [0, 1, 0, 1.]])
        X = one_hot_to_seq(x, alleles=np.array(['A', 'B']))
        assert(np.all(X == np.array(['AA', 'AB', 'BA', 'BB'])))
    
    def test_get_full_one_hot(self):
        X = get_full_space_one_hot(seq_length=2, n_alleles=2)
        assert(np.allclose(X,  [[1, 0, 1, 0], 
                                [0, 1, 1, 0],
                                [1, 0, 0, 1],
                                [0, 1, 0, 1]]))
        
        X = get_full_space_one_hot(seq_length=2, n_alleles=3)
        assert(np.allclose(X, [[1, 0, 0, 1, 0, 0],
                               [0, 1, 0, 1, 0, 0],
                               [0, 0, 1, 1, 0, 0],
                               [1, 0, 0, 0, 1, 0],
                               [0, 1, 0, 0, 1, 0],
                               [0, 0, 1, 0, 1, 0],
                               [1, 0, 0, 0, 0, 1],
                               [0, 1, 0, 0, 0, 1],
                               [0, 0, 1, 0, 0, 1],]))
    
    def test_get_full_binary(self):
        from epik.src.utils import get_full_space_binary
        X = get_full_space_binary(seq_length=2)
        assert(np.allclose(X, [[ 1, 1], 
                               [-1, 1],
                               [ 1,-1],
                               [-1,-1]]))
        
    
    def test_diploid_encoding(self):
        X = np.array(['00', '01', '11', '02', '22'])
        x = diploid_to_one_hot(X).numpy()
        assert(x.shape == (5, 2, 3))
        
        h0 = np.array([[1, 1], [1, 0], [0, 0], [1, 0], [0, 0]])
        ht = np.array([[0, 0], [0, 1], [1, 1], [0, 0], [0, 0]])
        h1 = np.array([[0, 0], [0, 0], [0, 0], [0, 1], [1, 1]])
        y = np.stack([h0, ht, h1], axis=2)
        assert(np.allclose(x, y))

    def test_calc_decay_rates(self):
        logit_rho = np.array([[0.],
                              [-0.69],
                              [0.69]])
        log_p = np.full((3, 3), 1/3.)
        decay_rates = calc_decay_rates(logit_rho, log_p,
                                       alleles=['A', 'B', 'C'],
                                       positions=[10, 12, 15])
        
        assert(np.all(decay_rates.columns == ['A', 'B', 'C']))
        assert(np.all(decay_rates.index == [10, 12, 15]))

        rho = np.exp(logit_rho) / (1 + np.exp(logit_rho))
        expected_decay_rates = (1 - (1 - rho) / (1 + 2 * rho)).flatten()
        decay_rates = decay_rates.mean(1).values.flatten()
        assert(np.allclose(decay_rates, expected_decay_rates))

        # With sqrt
        decay_rates = calc_decay_rates(logit_rho, log_p, sqrt=True,
                                       alleles=['A', 'B', 'C'],
                                       positions=[10, 12, 15])
        expected_decay_rates = (1 - np.sqrt((1 - rho) / (1 + 2 * rho))).flatten()
        decay_rates = decay_rates.mean(1).values.flatten()
        assert(np.allclose(decay_rates, expected_decay_rates))
    
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'UtilsTests']
    unittest.main()
