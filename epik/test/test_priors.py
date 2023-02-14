#!/usr/bin/env python
import unittest

import numpy as np
import torch

from epik.src.priors import AllelesProbPrior, LambdasExpDecayPrior


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
    
    def test_exp_decay_lambdas_prior(self):
        l = 5
        prior = LambdasExpDecayPrior(l, tau=0.2, train=False)
        log_lambdas = -torch.arange(l).to(dtype=torch.float)
        theta = prior.log_lambdas_to_theta(log_lambdas).detach()
        exp_theta = np.zeros(l)
        exp_theta[1] = -1
        assert(np.allclose(theta.numpy(), exp_theta))
        
        log_lambdas_star = prior.theta_to_log_lambdas(theta).detach().numpy()
        assert(np.allclose(log_lambdas, log_lambdas_star))
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'PriorTests']
    unittest.main()
