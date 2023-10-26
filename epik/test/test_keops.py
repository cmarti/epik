#!/usr/bin/env python
import unittest

import numpy as np
import torch

from epik.src.keops import (RhoPiKernel, RhoKernel)
from epik.src.utils import get_full_space_one_hot
from pykeops.torch import LazyTensor



class KernelsTests(unittest.TestCase):
    def test_rho_kernel(self):
        l, a = 1, 2
        I = torch.eye(a ** l)
        
        logit_rho = torch.tensor([[0.]])
        rho = torch.exp(logit_rho) / (1 + torch.exp(logit_rho))
        exp_d0 = torch.prod(1 + rho).numpy()
        
        kernel = RhoKernel(n_alleles=a, seq_length=l, logit_rho0=logit_rho)
        x = get_full_space_one_hot(l, a)
        
        cov = kernel._nonkeops_forward(x, x).detach().numpy()
        assert(np.allclose(cov[0, 0], exp_d0))
        assert(cov[0, 1] == 0.50)
        
        cov = (kernel._keops_forward(x, x) @ I).detach().numpy()
        assert(np.allclose(cov[0, 0], exp_d0))
        assert(cov[0, 1] == 0.50)
        
        l, a = 2, 2
        I = torch.eye(a ** l)
        logit_rho = torch.tensor([[0.], [0.]])
        rho = torch.exp(logit_rho) / (1 + torch.exp(logit_rho))
        exp_d0 = torch.prod(1 + rho)
        
        kernel = RhoKernel(n_alleles=a, seq_length=l, logit_rho0=logit_rho)
        x = get_full_space_one_hot(l, a)
        
        cov = kernel._nonkeops_forward(x, x).detach().numpy()
        d0, d1, d2 = cov[0, 0], cov[0, 1], cov[0, 3]
        assert(np.allclose(d0, exp_d0))
        assert(np.allclose(d1 / d0, d2 / d1))
        
        cov = (kernel._keops_forward(x, x) @ I).detach().numpy()
        d0, d1, d2 = cov[0, 0], cov[0, 1], cov[0, 3]
        assert(np.allclose(d0, exp_d0))
        assert(np.allclose(d1 / d0, d2 / d1))
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'KernelsTests']
    unittest.main()
