#!/usr/bin/env python
import unittest

import numpy as np
import matplotlib.pyplot as plt
import torch

from epik.src.priors import LambdasExpDecayPrior, AllelesProbPrior
from epik.src.utils import seq_to_one_hot, get_tensor
from epik.src.settings import TEST_DATA_DIR
from os.path import join


class PriorTests(unittest.TestCase):
    def test_alleles_prior_full(self):
        for a in range(2, 5):
            for l in range(2, 5):
                prior = AllelesProbPrior(l, a)
                logp = prior.get_logp0()
                assert(np.allclose(logp, 0))
                assert(logp.shape == (l, a+1))
                
                # Remains unchanged
                logp = prior.resize_logp(logp)
                assert(np.allclose(logp, 0))
                assert(logp.shape == (l, a+1))
                
                # Normalize and get probabilities
                norm_logp = prior.normalize_logp(logp)
                p = torch.exp(norm_logp)
                assert(np.allclose(p, 1./(a + 1)))
                
                # Calculate betas (log odds)
                beta = prior.norm_logp_to_beta(norm_logp)
                assert(np.allclose(beta, np.log(a)))
    
    def test_alleles_prior_partial(self):
        for a in range(2, 5):
            for l in range(2, 5):

                # Same weight across alleles
                prior = AllelesProbPrior(l, a, alleles_equal=True)
                logp = prior.get_logp0()
                assert(np.allclose(logp, 0))
                assert(logp.shape == (l, 1))
        
                logp = prior.resize_logp(logp)
                assert(np.allclose(logp, 0))
                assert(logp.shape == (l, a+1))
                
                # Same weight across sites
                prior = AllelesProbPrior(l, a, sites_equal=True)
                logp = prior.get_logp0()
                assert(np.allclose(logp, 0))
                assert(logp.shape == (1, a+1))
                
                logp = prior.resize_logp(logp)
                assert(np.allclose(logp, 0))
                assert(logp.shape == (l, a+1))
                
                # Same weight across sites and alleles
                prior = AllelesProbPrior(l, a, sites_equal=True, alleles_equal=True)
                logp = prior.get_logp0()
                assert(np.allclose(logp, 0))
                assert(logp.shape == (1, 1))
                
                logp = prior.resize_logp(logp)
                assert(np.allclose(logp, 0))
                assert(logp.shape == (l, a+1))
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'PriorTests']
    unittest.main()
