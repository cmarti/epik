#!/usr/bin/env python
import unittest

import numpy as np
import torch

from epik.src.keops import (RhoPiKernel, RhoKernel, VarianceComponentKernel,
                            AdditiveKernel)
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
        
    def test_vc_kernel(self):
        l, a = 2, 2
        x = get_full_space_one_hot(l, a)
        
        # Constant kernel
        log_lambdas0 = torch.log(torch.tensor([1, 0, 0]))
        kernel = VarianceComponentKernel(n_alleles=a, seq_length=l,
                                         log_lambdas0=log_lambdas0)
        cov0 = kernel._nonkeops_forward(x, x).detach().numpy()
        assert(np.allclose(cov0, 0.25))
        
        # k=1        
        log_lambdas0 = torch.log(torch.tensor([0, 1, 0]))
        kernel = VarianceComponentKernel(n_alleles=a, seq_length=l,
                                         log_lambdas0=log_lambdas0)
        cov1 = kernel._nonkeops_forward(x, x).detach().numpy()
        k1 = np.array([[2, 0, 0, -2],
                       [0, 2, -2, 0],
                       [0, -2, 2, 0],
                       [-2, 0, 0, 2]], dtype=np.float32)
        assert(np.allclose(cov1,  k1 / 4, atol=1e-4))
        
        # k=2
        log_lambdas0 = torch.log(torch.tensor([0, 0, 1]))
        kernel = VarianceComponentKernel(n_alleles=a, seq_length=l,
                                         log_lambdas0=log_lambdas0)
        cov2 = kernel._nonkeops_forward(x, x).detach().numpy()
        k2 = np.array([[1, -1, -1, 1],
                       [-1, 1, 1, -1],
                       [-1, 1, 1, -1],
                       [1, -1, -1, 1]], dtype=np.float32)
        assert(np.allclose(cov2,  k2 / 4, atol=1e-4))
        
        # Using KeOops
        I = torch.eye(a ** l)
        log_lambdas0 = torch.log(torch.tensor([1, 0.5, 0.25]))
        kernel = VarianceComponentKernel(n_alleles=a, seq_length=l,
                                         log_lambdas0=log_lambdas0)
        cov1 = kernel._nonkeops_forward(x, x).detach().numpy()
        cov2 = (kernel._keops_forward(x, x) @ I).detach().numpy()
        cov3 = kernel._nonkeops_forward_polynomial_d(x, x).detach().numpy()
        assert(np.allclose(cov1, cov2))
        assert(np.allclose(cov1, cov3))
        
    def test_het_rbf_kernel(self):
        l, a = 2, 2
        x = get_full_space_one_hot(l, a)
        
        # Constant kernel
        log_ds0 = torch.log(torch.tensor([1, 2, 1, 2]))
        kernel = HetRBFKernel(n_alleles=a, seq_length=l, log_ds0=log_ds0)
        cov1 = kernel._nonkeops_forward(x, x).detach().numpy()
        cov2 = kernel._keops_forward(x, x).detach().numpy()
        assert(np.allclose(cov1, cov2))
    
    def test_additive_kernel(self):
        l, a = 2, 2
        I = torch.eye(a ** l)
        x = get_full_space_one_hot(l, a)
        
        # Additive kernel
        log_lambdas0 = torch.tensor([-10., 0])
        kernel = AdditiveKernel(n_alleles=a, seq_length=l, log_lambdas0=log_lambdas0)
        cov1 = kernel._nonkeops_forward(x, x).detach().numpy()
        assert(np.allclose(cov1[0], [2, 0, 0, -2], atol=0.01))
        
        k = kernel._keops_forward(x, x)
        cov2 = (k @ I).detach().numpy()
        assert(np.allclose(cov1, cov2))
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'KernelsTests']
    unittest.main()
