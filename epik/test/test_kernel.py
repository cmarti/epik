#!/usr/bin/env python
import unittest
import numpy as np
import matplotlib.pyplot as plt
import torch

from os.path import join
from scipy.linalg import cholesky

from epik.src.settings import TEST_DATA_DIR
from epik.src.kernel.haploid import (VarianceComponentKernel, RhoPiKernel,
                                     RhoKernel, AdditiveKernel, ARDKernel, 
                                     PairwiseKernel,
                                     calc_d_powers_inverse, calc_vandermonde_inverse)
from epik.src.utils import (seq_to_one_hot, get_tensor, diploid_to_one_hot,
                            get_full_space_one_hot)
from epik.src.kernel.base import AdditiveHeteroskedasticKernel


class KernelsTests(unittest.TestCase):
    def test_additive_kernel(self):
        l, a = 1, 2
        x = get_full_space_one_hot(l, a)
        I = torch.eye(a ** l)
        
        # Additive kernel with lambdas1 = 0 should return a constant matrix
        log_lambdas0 = torch.tensor([0., -10.])
        kernel = AdditiveKernel(n_alleles=a, seq_length=l, log_lambdas0=log_lambdas0)
        cov = kernel.forward(x, x).detach().numpy()
        assert(np.allclose(cov, 1, atol=0.01))
        
        # Additive kernel with lambdas0 = 1 should return a purely additive cov
        log_lambdas0 = torch.tensor([-10., 0.])
        kernel = AdditiveKernel(n_alleles=a, seq_length=l, log_lambdas0=log_lambdas0)
        cov = kernel.forward(x, x).detach().numpy()
        assert(np.allclose(cov, np.array([[1, -1],
                                          [-1, 1]]), atol=0.01))
        
        # Additive kernel with lambdas = 1 should return the identity
        log_lambdas0 = torch.tensor([0., 0])
        kernel = AdditiveKernel(n_alleles=a, seq_length=l, log_lambdas0=log_lambdas0)
        cov = kernel.forward(x, x).detach().numpy()
        assert(np.allclose(cov, (a ** l) * I, atol=0.01))
        
        l, a = 2, 2
        x = get_full_space_one_hot(l, a)

        # Constant kernel
        log_lambdas0 = torch.tensor([0., -10])
        kernel = AdditiveKernel(n_alleles=a, seq_length=l, log_lambdas0=log_lambdas0)
        cov = kernel.forward(x, x).detach().numpy()
        assert(np.allclose(cov, 1, atol=0.01))
        
        # Additive kernel
        log_lambdas0 = torch.tensor([-10., 0])
        kernel = AdditiveKernel(n_alleles=a, seq_length=l, log_lambdas0=log_lambdas0)
        cov = kernel.forward(x, x).detach().numpy()
        assert(np.allclose(cov[0], [2, 0, 0, -2], atol=0.01))
        
        # Additive kernel with larger variance
        log_lambdas0 = torch.tensor([-10., np.log(2)]).to(dtype=torch.float32)
        kernel = AdditiveKernel(n_alleles=a, seq_length=l, log_lambdas0=log_lambdas0)
        cov = kernel.forward(x, x).detach().numpy()
        assert(np.allclose(cov[0], [4, 0, 0, -4], atol=0.01))

        # Binary encoding
        x = torch.tensor([[1., 1],
                          [-1, 1],
                          [1, -1],
                          [-1, -1]])
        kernel = AdditiveKernel(n_alleles=a, seq_length=l, log_lambdas0=log_lambdas0, binary=True)
        cov = kernel.forward(x, x).detach().numpy()
        assert(np.allclose(cov[0], [4, 0, 0, -4], atol=0.01))
    
    def test_pairwise_kernel(self):
        l, a = 2, 2
        x = get_full_space_one_hot(l, a)

        # Constant kernel
        log_lambdas0 = torch.tensor([0., -10, -10])
        kernel = PairwiseKernel(n_alleles=a, seq_length=l, log_lambdas0=log_lambdas0)
        cov = kernel.forward(x, x).detach().numpy()
        assert(np.allclose(cov, 1, atol=0.01))
        
        cov2 = kernel._keops_forward(x, x).to_dense().detach().numpy()
        assert(np.allclose(cov2, cov, atol=0.01))
        
        # Additive kernel
        log_lambdas0 = torch.tensor([-10., 0, -10.])
        kernel = PairwiseKernel(n_alleles=a, seq_length=l, log_lambdas0=log_lambdas0)
        cov = kernel.forward(x, x).detach().numpy()
        assert(np.allclose(cov[0], [2, 0, 0, -2], atol=0.01))

        cov2 = kernel._keops_forward(x, x).to_dense().detach().numpy()
        assert(np.allclose(cov2, cov, atol=0.01))
        
        # Additive kernel with larger variance
        log_lambdas0 = torch.tensor([-10., np.log(2), -10.]).to(dtype=torch.float32)
        kernel = PairwiseKernel(n_alleles=a, seq_length=l, log_lambdas0=log_lambdas0)
        cov = kernel.forward(x, x).detach().numpy()
        assert(np.allclose(cov[0], [4, 0, 0, -4], atol=0.01))

        cov2 = kernel._keops_forward(x, x).to_dense().detach().numpy()
        assert(np.allclose(cov2, cov, atol=0.01))

        # Verify pairwise model with VC kernel
        log_lambdas0 = torch.tensor([1., np.log(2), -1.]).to(dtype=torch.float32)
        kernel1 = PairwiseKernel(n_alleles=a, seq_length=l, log_lambdas0=log_lambdas0)
        cov1 = kernel1.forward(x, x).detach().numpy()
        kernel2 = VarianceComponentKernel(n_alleles=a, seq_length=l, log_lambdas0=log_lambdas0)
        cov2 = kernel2.forward(x, x).detach().numpy()
        assert(np.allclose(cov1, cov2, atol=0.01))

        # Binary encoding
        x = torch.tensor([[1., 1],
                          [-1, 1],
                          [1, -1],
                          [-1, -1]])
        kernel = PairwiseKernel(n_alleles=a, seq_length=l, log_lambdas0=log_lambdas0, binary=True)
        cov3 = kernel._nonkeops_forward(x, x).detach().numpy()
        assert(np.allclose(cov1, cov3, atol=0.01))

        cov4 = kernel._keops_forward(x, x).detach().numpy()
        assert(np.allclose(cov1, cov4, atol=0.01))

        # Test in long sequences
        a, l, n = 2, 100, 3
        I = torch.eye(n)
        x = np.random.choice([0, 1], size=n * l)
        x = np.vstack([x, 1- x]).T.reshape((n, 2 * l))
        x = torch.tensor(x, dtype=torch.float32)
        
        log_lambdas0 = torch.tensor([-10., 0, -30.]).to(dtype=torch.float32)
        kernel = PairwiseKernel(n_alleles=a, seq_length=l, log_lambdas0=log_lambdas0)
        cov = kernel._nonkeops_forward(x, x).detach().numpy()
        cov2 = kernel._keops_forward(x, x).to_dense().detach().numpy()
        assert(np.allclose(cov2, cov, atol=1e-4))
    
    def test_d_powers_coeffs(self):
        l = 3
        
        d = torch.arange(l+1).unsqueeze(0).to(dtype=torch.float64)
        A = d.T ** d
        m1 = torch.linalg.inv(A)
        print(m1)

        m2 = calc_d_powers_inverse(l)
        print(m2)

        m3 = calc_vandermonde_inverse(np.arange(l+1))
        print(m3)

        for l in range(10):
            m2 = calc_d_powers_inverse(l)
        print(m2)

    def test_vc_kernel(self):
        l, a = 2, 2
        x = get_full_space_one_hot(l, a)

        # k=0        
        log_lambdas0 = torch.tensor([0, -20., -20.], dtype=torch.float32)
        kernel = VarianceComponentKernel(n_alleles=a, seq_length=l,
                                         log_lambdas0=log_lambdas0)
        cov = kernel.forward(x, x).detach().numpy()
        assert(np.allclose(cov, 1, atol=1e-4))
        
        I = torch.eye(a ** l)
        cov2 = (kernel._keops_forward(x, x) @ I).detach().numpy()
        assert(np.allclose(cov2, cov, atol=1e-4))
        
        # k=1        
        log_lambdas0 = torch.tensor([-10., 0., -10.], dtype=torch.float32)
        kernel = VarianceComponentKernel(n_alleles=a, seq_length=l,
                                         log_lambdas0=log_lambdas0)
        cov = kernel.forward(x, x).detach().numpy()
        k1 = np.array([[1, 0, 0, -1],
                       [0, 1, -1, 0],
                       [0, -1, 1, 0],
                       [-1, 0, 0, 1]], dtype=np.float32)
        assert(np.allclose(cov,  k1 * 2, atol=1e-4))
        
        I = torch.eye(a ** l)
        cov2 = (kernel._keops_forward(x, x) @ I).detach().numpy()
        assert(np.allclose(cov2, cov, atol=1e-4))
        
        # k=2
        log_lambdas0 = torch.tensor([-20., -20., 0.], dtype=torch.float32)
        kernel = VarianceComponentKernel(n_alleles=a, seq_length=l,
                                         log_lambdas0=log_lambdas0)
        cov = kernel.forward(x, x).detach().numpy()
        k2 = np.array([[1, -1, -1, 1],
                       [-1, 1, 1, -1],
                       [-1, 1, 1, -1],
                       [1, -1, -1, 1]], dtype=np.float32)
        assert(np.allclose(cov,  k2, atol=1e-4))
        
        I = torch.eye(a ** l)
        cov2 = (kernel._keops_forward(x, x) @ I).detach().numpy()
        assert(np.allclose(cov2, cov))
    
    def test_vc_kernel_max_k(self):
        l, a = 2, 2
        I = torch.eye(a ** l)
        x = get_full_space_one_hot(l, a)

        # Constant kernel
        log_lambdas0 = torch.tensor([0., -10])
        kernel = VarianceComponentKernel(n_alleles=a, seq_length=l,
                                         log_lambdas0=log_lambdas0, max_k=1)
        cov = kernel.forward(x, x).detach().numpy()
        assert(np.allclose(cov, 1, atol=0.01))
        
        cov2 = (kernel._keops_forward(x, x) @ I).detach().numpy()
        assert(np.allclose(cov2, cov, atol=0.01))
        
        # Additive kernel
        log_lambdas0 = torch.tensor([-20., 0])
        kernel = VarianceComponentKernel(n_alleles=a, seq_length=l,
                                         log_lambdas0=log_lambdas0, max_k=1)
        cov = kernel.forward(x, x).detach().numpy()
        assert(np.allclose(cov[0], [2, 0, 0, -2], atol=0.01))

        cov2 = (kernel._keops_forward(x, x) @ I).detach().numpy()
        assert(np.allclose(cov2, cov, atol=0.01))
        
        # Additive kernel with larger variance
        log_lambdas0 = torch.tensor([-10., np.log(2)]).to(dtype=torch.float32)
        kernel = VarianceComponentKernel(n_alleles=a, seq_length=l,
                                         log_lambdas0=log_lambdas0, max_k=1)
        cov = kernel.forward(x, x).detach().numpy()
        assert(np.allclose(cov[0], [4, 0, 0, -4], atol=0.01))

        cov2 = (kernel._keops_forward(x, x) @ I).detach().numpy()
        assert(np.allclose(cov2, cov, atol=0.01))

        # Test in long sequences
        a, l, n = 2, 100, 3
        I = torch.eye(n)
        x = np.random.choice([0, 1], size=n * l)
        x = np.vstack([x, 1- x]).T.reshape((n, 2 * l))
        x = torch.tensor(x, dtype=torch.float32)
        
        log_lambdas0 = torch.tensor([-10., 1, -30.]).to(dtype=torch.float32)
        kernel = VarianceComponentKernel(n_alleles=a, seq_length=l, max_k=2,
                                         log_lambdas0=log_lambdas0)
        cov = kernel._nonkeops_forward(x, x).detach().numpy()
        cov2 = (kernel._keops_forward(x, x) @ I).detach().numpy()
        assert(np.allclose(cov2, cov, atol=1e-4))

    def test_vc_kernel_bigger(self):
        l, a = 6, 4
        I = torch.eye(a ** l)
        x = get_full_space_one_hot(l, a)

        kernel = VarianceComponentKernel(n_alleles=a, seq_length=l)
        cov1 = kernel._nonkeops_forward_polynomial_d(x, x).detach().numpy()
        cov2 = kernel._nonkeops_forward_hamming_class(x, x).detach().numpy()
        cov3 = (kernel._keops_forward(x, x) @ I).detach().numpy()
        w_d = kernel.get_w_d().detach().numpy()
        
        assert(np.allclose(np.unique(w_d)[1:], np.unique(cov1)[1:], atol=1e-3))
        assert(np.allclose(np.unique(w_d)[1:], np.unique(cov2)[1:], atol=1e-3))
        assert(np.allclose(np.unique(w_d)[1:], np.unique(cov3)[1:], atol=1e-3))
        
    def test_rho_kernel(self):
        l, a = 1, 2
        logit_rho0 = torch.tensor([0.])
        kernel = RhoKernel(n_alleles=a, seq_length=l, logit_rho0=logit_rho0)
        x = get_full_space_one_hot(l, a)
        cov = kernel.forward(x, x).detach().numpy()
        assert(cov[0, 0] == 1.5)
        assert(cov[0, 1] == 0.5)
        
        I = torch.eye(a ** l)
        cov2 = (kernel._keops_forward(x, x) @ I).detach().numpy()
        assert(np.allclose(cov2, cov))
        
        l, a = 2, 2
        logit_rho0 = torch.tensor([[0.], [0.]])
        rho = np.exp(logit_rho0.numpy()) / (1 + np.exp(logit_rho0.numpy()))
        kernel = RhoKernel(n_alleles=a, seq_length=l, logit_rho0=logit_rho0)
        x = get_full_space_one_hot(l, a)
        cov = kernel.forward(x, x).detach().numpy()
        assert(np.allclose(cov[0, :], [1.5 ** 2, 1.5 * 0.5, 1.5 * 0.5, 0.5 ** 2]))

        I = torch.eye(a ** l)
        cov2 = (kernel._keops_forward(x, x) @ I).detach().numpy()
        assert(np.allclose(cov, cov))

        logit_rho0 = torch.tensor([[0.], [-np.log(3)]], dtype=torch.float32)
        rho = (np.exp(logit_rho0.numpy()) / (1 + np.exp(logit_rho0.numpy()))).flatten()
        kernel = RhoKernel(n_alleles=a, seq_length=l, logit_rho0=logit_rho0)
        cov = kernel.forward(x, x).detach().numpy()
        expected = np.array([(1 + rho[0]) * (1 + rho[1]),
                             (1 - rho[0]) * (1 + rho[1]),
                             (1 + rho[0]) * (1 - rho[1]),
                             (1 - rho[0]) * (1 - rho[1])])
        assert(np.allclose(cov[0, :], expected))
        
        I = torch.eye(a ** l)
        cov2 = (kernel._keops_forward(x, x) @ I).detach().numpy()
        assert(np.allclose(cov2, cov))
    
    def test_rho_pi_kernel(self):
        l, a = 1, 2
        logit_rho0 = torch.tensor([[0.]])
        log_p0 = torch.tensor(np.log([[0.2, 0.8]]), dtype=torch.float32)
        kernel = RhoPiKernel(n_alleles=a, seq_length=l,
                             logit_rho0=logit_rho0, log_p0=log_p0)
        x = get_full_space_one_hot(l, a)
        cov = kernel.forward(x, x).detach().numpy()
        expected = np.array([[3, 0.5],
                             [0.5, 1.125]])
        assert(np.allclose(cov, expected))

        I = torch.eye(a ** l)
        cov2 = (kernel._keops_forward(x, x) @ I).detach().numpy()
        assert(np.allclose(cov2, cov))
        
        l, a = 2, 2
        logit_rho0 = torch.tensor([[0.],
                                   [0.]], dtype=torch.float32)
        log_p0 = torch.tensor(np.log([[0.2, 0.8],
                                      [0.5, 0.5]]), dtype=torch.float32)
        kernel = RhoPiKernel(n_alleles=a, seq_length=l,
                             logit_rho0=logit_rho0, log_p0=log_p0)
        x = get_full_space_one_hot(l, a)
        rho = np.array([0.5, 0.5])
        eta = np.array([[4., 0.25],
                        [1., 1.  ]])
        cov = kernel.forward(x, x).detach().numpy()
        expected = np.array([(1 + rho[0] * eta[0, 0]) * (1 + rho[1] * eta[1, 0]),
                             (1 - rho[0])             * (1 + rho[1] * eta[1, 0]),
                             (1 + rho[0] * eta[0, 0]) * (1 - rho[1]),
                             (1 - rho[0])             * (1 - rho[1])])
        assert(np.allclose(cov[0, :], expected))

        I = torch.eye(a ** l)
        cov2 = (kernel._keops_forward(x, x) @ I).detach().numpy()
        assert(np.allclose(cov2, cov))
        
    def test_ARD_kernel(self):
        l, a = 1, 2
        logit_rho0 = torch.tensor([[0.]])
        log_p0 = torch.tensor(np.log([[0.2, 0.8]]), dtype=torch.float32)
        kernel = ARDKernel(n_alleles=a, seq_length=l,
                           logit_rho0=logit_rho0, log_p0=log_p0, log_var0=0.)
        x = get_full_space_one_hot(l, a)
        cov = kernel.forward(x, x).detach().numpy()
        expected = np.array([[1, 0.5 / np.sqrt(3 * 1.125)],
                             [0.5 / np.sqrt(3 * 1.125), 1]])
        assert(np.allclose(cov, expected))

        I = torch.eye(a ** l)
        cov2 = (kernel._keops_forward(x, x) @ I).detach().numpy()
        assert(np.allclose(cov2, cov))
        
        l, a = 2, 2
        logit_rho0 = torch.tensor([[0.],
                                   [0.]], dtype=torch.float32)
        log_p0 = torch.tensor(np.log([[0.2, 0.8],
                                      [0.5, 0.5]]), dtype=torch.float32)
        kernel = ARDKernel(n_alleles=a, seq_length=l,
                             logit_rho0=logit_rho0, log_p0=log_p0, log_var0=0.)
        x = get_full_space_one_hot(l, a)
        rho = np.array([0.5, 0.5])
        eta = np.array([[4., 0.25],
                        [1., 1.  ]])
        cov = kernel.forward(x, x).detach().numpy()
        expected = np.array([1, 
                             (1 - rho[0]) / np.sqrt((1 + rho[0] * eta[0, 0]) * (1 + rho[0] * eta[0, 1])),
                             (1 - rho[1]) / np.sqrt((1 + rho[1] * eta[1, 0]) * (1 + rho[1] * eta[1, 1])),
                             np.nan])
        expected[3] = expected[1] * expected[2]
        assert(np.allclose(cov[0, :], expected))
        
        I = torch.eye(a ** l)
        cov2 = (kernel._keops_forward(x, x) @ I).detach().numpy()
        assert(np.allclose(cov2, cov))
        
    def test_heteroskedastic_kernel(self):
        l, a = 1, 2
        x = get_full_space_one_hot(l, a)
        
        logit_rho0 = torch.tensor([[0.]])
        kernel = RhoKernel(n_alleles=a, seq_length=l,
                           logit_rho0=logit_rho0)
        cov1 = kernel.forward(x, x)
        assert(cov1[0, 0] == 1.5)
        assert(cov1[0, 1] == 0.5)
        
        kernel = AdditiveHeteroskedasticKernel(kernel)
        cov2 = kernel.forward(x, x)
        assert(cov2[0, 0] < 1.5)
        assert(cov2[0, 1] < 0.5)


class OldKernelsTests(unittest.TestCase):
    def test_diploid_kernel(self):
        kernel = DiploidKernel()
        X = ['0', '1', '2']
        x = diploid_to_one_hot(X)
        
        # Purely constant function
        mu, lda, eta = 1, 0, 0
        cov = kernel._forward(x, x, mu, lda, eta).numpy()
        assert(np.allclose(cov, 1))
        
        # Purely additive function
        mu, lda, eta = 0, 1, 0
        cov = kernel._forward(x, x, mu, lda, eta).numpy()
        exp = np.array([[ 2.,  0., -2.],
                        [ 0.,  0.,  0.],
                        [-2.,  0.,  2.]])
        assert(np.allclose(cov, exp))
        
        # Purely dominant function
        mu, lda, eta = 0, 0, 1
        cov = kernel._forward(x, x, mu, lda, eta).numpy()
        exp = np.array([[ 1,  -1,   1.],
                        [-1.,  1., -1.],
                        [ 1., -1.,  1.]])
        assert(np.allclose(cov, exp))
        
    def test_skewed_vc_kernel(self):
        l, a = 2, 2
        lambdas_prior = LambdasExpDecayPrior(l, tau=0.2)
        p_prior = AllelesProbPrior(l, a)
        kernel = SkewedVCKernel(a, l, lambdas_prior, p_prior,
                                dtype=torch.float32)
        
        logp = torch.tensor([[0, 0, -10], [0, 0, -10]], dtype=torch.float32)
        logp = p_prior.normalize_logp(logp)
        x = torch.tensor([[1, 0, 1, 0],
                          [0, 1, 1, 0],
                          [1, 0, 0, 1],
                          [0, 1, 0, 1]], dtype=torch.float32)

        # k=0        
        lambdas = torch.tensor([1, 0, 0], dtype=torch.float32)
        cov = kernel._forward(x, x, lambdas, logp)
        assert(np.allclose(cov, 1))
         
        # k=1        
        lambdas = torch.tensor([0, 1, 0], dtype=torch.float32)
        cov = kernel._forward(x, x, lambdas, logp).numpy()
        k1 = np.array([[2, 0, 0, -2],
                       [0, 2, -2, 0],
                       [0, -2, 2, 0],
                       [-2, 0, 0, 2]], dtype=np.float32)
        assert(np.abs(cov - k1).mean() < 1e-4)
         
        # k=2
        lambdas = torch.tensor([0, 0, 1], dtype=torch.float32)
        cov = kernel._forward(x, x, lambdas, logp).numpy()
        k2 = np.array([[1, -1, -1, 1],
                       [-1, 1, 1, -1],
                       [-1, 1, 1, -1],
                       [1, -1, -1, 1]], dtype=np.float32)
        assert(np.abs(cov - k2).mean() < 1e-4)
    
        # Longer seqs
        l, a = 6, 2
        lambdas_prior = LambdasExpDecayPrior(l, tau=0.2)
        p_prior = AllelesProbPrior(l, a)
        kernel = SkewedVCKernel(a, l, lambdas_prior, p_prior,
                                dtype=torch.float32)
        
        logp = torch.tensor([[0, 0, -10]] * l, dtype=torch.float32)
        logp = p_prior.normalize_logp(logp)
        x = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                          [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                          [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
                          [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
                          [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
                          [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
                          [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]], dtype=torch.float32)
        
        lambdas = torch.tensor([0, 1, 0, 0, 0, 0, 0], dtype=torch.float32)
        cov = kernel._forward(x[:1], x[1:], lambdas, logp).numpy()
        assert(np.allclose(cov[0][:-1] - cov[0][1:], 2, atol=0.05))
        
        # More than 2 alleles
        l, a = 2, 3
        lambdas_prior = LambdasExpDecayPrior(l, tau=0.2)
        p_prior = AllelesProbPrior(l, a)
        kernel = SkewedVCKernel(a, l, lambdas_prior, p_prior,
                                dtype=torch.float32)
        logp = torch.tensor([[0, 0, 0, -10]] * l, dtype=torch.float32)
        logp = p_prior.normalize_logp(logp)
        x = torch.tensor([[1, 0, 0, 1, 0, 0],
                          [0, 1, 0, 1, 0, 0],
                          [0, 0, 1, 1, 0, 0],
                          [1, 0, 0, 0, 1, 0],
                          [0, 1, 0, 0, 1, 0],
                          [0, 0, 1, 0, 1, 0],
                          [1, 0, 0, 0, 0, 1],
                          [0, 1, 0, 0, 0, 1],
                          [0, 0, 1, 0, 0, 1]], dtype=torch.float32)
        
        lambdas = torch.tensor([0, 1, 0], dtype=torch.float32)
        cov = kernel._forward(x[:1], x, lambdas, logp).numpy()
        k1 = np.array([4, 1, 1, 1, -2, -2, 1, -2, -2])
        assert(np.abs(cov[0] - k1).mean() < 1e-4)
    
    def test_skewed_vc_kernel_different_qs(self):
        for q in np.linspace(0.1, 0.9, 11):
            kernel = SkewedVCKernel(n_alleles=2, seq_length=2, q=q, dtype=torch.float32)
            logp = torch.log(torch.tensor([[0.5, 0.5, 0.00001],
                                            [0.5, 0.5, 0.00001]], dtype=torch.float32))
            x = torch.tensor([[1, 0, 1, 0],
                              [0, 1, 1, 0],
                              [1, 0, 0, 1],
                              [0, 1, 0, 1]], dtype=torch.float32)
    
            # k=0        
            lambdas = torch.tensor([1, 0, 0], dtype=torch.float32)
            cov = kernel._forward(x, x, lambdas, logp).numpy()
            assert(np.allclose(cov, 1, atol=1e-4))
            
            # k=1        
            lambdas = torch.tensor([0, 1, 0], dtype=torch.float32)
            cov = kernel._forward(x, x, lambdas, logp).numpy()
            k1 = np.array([[2, 0, 0, -2],
                           [0, 2, -2, 0],
                           [0, -2, 2, 0],
                           [-2, 0, 0, 2]], dtype=np.float32)
            assert(np.abs(cov - k1).mean() < 1e-4)
            
            # k=2
            lambdas = torch.tensor([0, 0, 1], dtype=torch.float32)
            cov = kernel._forward(x, x, lambdas, logp).numpy()
            k2 = np.array([[1, -1, -1, 1],
                           [-1, 1, 1, -1],
                           [-1, 1, 1, -1],
                           [1, -1, -1, 1]], dtype=np.float32)
            assert(np.abs(cov - k2).mean() < 1e-4)
    
    def test_skewed_vc_kernel_gpu(self):
        gpu = torch.device('cuda:0')
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2, dtype=torch.float32).to(gpu)
        logp = torch.log(torch.tensor([[0.5, 0.5, 0.00001],
                                        [0.5, 0.5, 0.00001]], dtype=torch.float32)).to(gpu)
        x = torch.tensor([[1, 0, 1, 0],
                          [0, 1, 1, 0],
                          [1, 0, 0, 1],
                          [0, 1, 0, 1]], dtype=torch.float32)
        xgpu = x.to(gpu)

        # k=0        
        lambdas = torch.tensor([1, 0, 0], dtype=torch.float32).to(gpu)
        cov = kernel._forward(xgpu, xgpu, lambdas, logp).cpu().numpy()
        assert(np.allclose(cov, 1))
        
        # k=1        
        lambdas = torch.tensor([0, 1, 0], dtype=torch.float32).to(gpu)
        cov = kernel._forward(xgpu, xgpu, lambdas, logp).cpu().numpy()
        k1 = np.array([[2, 0, 0, -2],
                       [0, 2, -2, 0],
                       [0, -2, 2, 0],
                       [-2, 0, 0, 2]], dtype=np.float32)
        assert(np.abs(cov - k1).mean() < 1e-4)
        
        # k=2
        lambdas = torch.tensor([0, 0, 1], dtype=torch.float32).to(gpu)
        cov = kernel._forward(xgpu, xgpu, lambdas, logp).cpu().numpy()
        k2 = np.array([[1, -1, -1, 1],
                       [-1, 1, 1, -1],
                       [-1, 1, 1, -1],
                       [1, -1, -1, 1]], dtype=np.float32)
        assert(np.abs(cov - k2).mean() < 1e-4)
        
        # Test now with the public method
        logp = torch.log(torch.tensor([[0.5, 0.5, 0.00001],
                                        [0.5, 0.5, 0.00001]], dtype=torch.float32))
        log_lambdas = torch.tensor([0, -10], dtype=torch.float32)

        for q in np.linspace(0.1, 0.9, 11):
            kernel = SkewedVCKernel(n_alleles=2, seq_length=2, q=q,
                                    log_lambdas0=log_lambdas,
                                    starting_p=torch.exp(logp),
                                    dtype=torch.float32).to(gpu)
            cov_gpu = kernel.forward(xgpu, xgpu).detach().cpu().numpy()
            
            kernel = SkewedVCKernel(n_alleles=2, seq_length=2, q=q,
                                    log_lambdas0=log_lambdas,
                                    starting_p=torch.exp(logp),
                                    dtype=torch.float32)
            cov_cpu = kernel.forward(x, x).detach().numpy()
            assert(np.allclose(cov_gpu, cov_cpu, atol=1e-3))
            assert(np.allclose(cov_gpu, k1, atol=1e-3))
        
    def test_skewed_vc_kernel_diag(self):
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2, dtype=torch.float32)
        logp = torch.log(torch.tensor([[0.45, 0.45, 0.1],
                                        [0.45, 0.45, 0.1]], dtype=torch.float32))
        x = torch.tensor([[1, 0, 1, 0],
                          [0, 1, 1, 0],
                          [1, 0, 0, 1],
                          [0, 1, 0, 1]], dtype=torch.float32)

        lambdas = torch.tensor([1, 0.5, 0.1], dtype=torch.float32)
        cov = kernel._forward(x, x, lambdas, logp, diag=False)
        var = kernel._forward(x, x, lambdas, logp, diag=True)
        assert(np.allclose(cov.numpy().diagonal(), var.numpy()))
        
    def test_kernel_params(self):
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2,
                                train_p=False, train_lambdas=False,
                                log_lambdas0=np.array([0, -10]),
                                dtype=torch.float32)
        x = torch.tensor([[1, 0, 1, 0],
                          [0, 1, 1, 0],
                          [1, 0, 0, 1],
                          [0, 1, 0, 1]], dtype=torch.float32)
        cov = kernel.forward(x, x)
        assert(np.allclose(cov[0], [2, 0, 0, -2], atol=0.01))
        
    def test_vc_kernels_variable_lengths(self):
        for alleles in [['A', 'B'], ['A', 'B', 'C']]:
            alpha = len(alleles)
            for l in range(2, 8):
                seqs = np.array(['A' * i + 'B' * (l-i) for i in range(l)])
                train_x = seq_to_one_hot(seqs, alleles=alleles)
                
                for i in range(l):
                    log_lambdas0 = -10 * np.ones(l)
                    log_lambdas0[i] = 0
                 
                    ker = VCKernel(alpha, l, tau=.1,
                                   log_lambdas0=log_lambdas0)
                    cov1 = ker.forward(train_x, train_x).detach().numpy()
                    
                    ker = SkewedVCKernel(alpha, l, q=0.7, tau=.1,
                                         log_lambdas0=log_lambdas0,
                                         dtype=torch.float32,
                                         lambdas_prior='2nd_order_diff')                    
                    cov2 = ker.forward(train_x, train_x).detach().numpy()
                    
                    # Tests they are all similar
                    logfc = np.nanmean(np.log2(cov1 / cov2))
                    assert(np.allclose(logfc, 0, atol=1))
                    
    def test_vc_kernels_variable_lengths_gpu(self):
        gpu = torch.device('cuda:0')
        for alleles in [['A', 'B'], ['A', 'B', 'C']]:
            alpha = len(alleles)
            for l in range(2, 8):
                seqs = np.array(['A' * i + 'B' * (l-i) for i in range(l)])
                train_x = seq_to_one_hot(seqs, alleles=alleles)
                train_x_gpu = train_x.to(gpu)
                
                for i in range(l):
                    log_lambdas0 = -10 * np.ones(l)
                    log_lambdas0[i] = 0
                    log_lambdas0 = get_tensor(log_lambdas0)
                 
                    # CPU
                    ker = VCKernel(alpha, l, tau=.1,
                                   log_lambdas0=log_lambdas0)
                    cov1 = ker.forward(train_x, train_x).detach().numpy()
                    
                    ker = SkewedVCKernel(alpha, l, q=0.7, tau=.1,
                                         log_lambdas0=log_lambdas0)                    
                    cov2 = ker.forward(train_x, train_x).detach().numpy()
                    
                    # GPU
                    ker = VCKernel(alpha, l, tau=.1,
                                   log_lambdas0=log_lambdas0).to(gpu)
                    cov3 = ker.forward(train_x_gpu, train_x_gpu).detach().cpu().numpy()
                     
                    ker = SkewedVCKernel(alpha, l, tau=.1, q=0.7,
                                         log_lambdas0=log_lambdas0).to(gpu)
                    cov4 = ker.forward(train_x_gpu, train_x_gpu).detach().cpu().numpy()
                    
                    # Tests they are all similar
                    logfc = np.nanmean(np.log2(cov1 / cov2))
                    assert(np.allclose(logfc, 0, atol=1e-2))
                    
                    logfc = np.nanmean(np.log2(cov1 / cov3))
                    assert(np.allclose(logfc, 0, atol=1e-2))
                     
                    logfc = np.nanmean(np.log2(cov1 / cov4))
                    assert(np.allclose(logfc, 0, atol=1e-2))

    def test_plot_vc_cov_d_function(self):
        fig, axes = plt.subplots(1, 1, figsize=(5, 4))
        
        alleles = ['A', 'B']
        alpha = len(alleles)
        for l in range(2, 8):
            seqs = np.array(['A' * i + 'B' * (l-i) for i in range(l)])
            train_x = seq_to_one_hot(seqs, alleles=alleles).to(torch.float64)
            
            i = 0
            log_lambdas0 = -10 * np.ones(l)
            log_lambdas0[i] = 0
         
            ker = SkewedVCKernel(alpha, l, q=0.7, tau=.1,
                                 log_lambdas0=log_lambdas0,
                                 dtype=torch.float64)
            cov1 = ker.forward(train_x, train_x).detach().numpy()
            
            ker = VCKernel(alpha, l, tau=.1,
                           log_lambdas0=log_lambdas0)
            cov2 = ker.forward(train_x, train_x).detach().numpy()
            
            axes.plot(cov1[0], label='sVC({})'.format(l))
            axes.plot(cov2[0], label='VC({})'.format(l), linestyle='--')
            
            logfc = np.nanmean(np.log2(cov1 / cov2))
            assert(np.allclose(logfc, 0, atol=1e-2))
            
        axes.legend()
        axes.set(xlabel='Hamming distance', ylabel='Additive covariance')
        fig.savefig(join(TEST_DATA_DIR, 'cov_dist.png'))
    
    def xtest_calc_polynomial_coeffs(self):
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2, dtype=torch.float32)
        lambdas = get_tensor(kernel.calc_eigenvalues())
        V = torch.stack([torch.pow(lambdas, i) for i in range(3)], 1)

        B = kernel.coeffs
        P = torch.matmul(B, V).numpy()
        assert(np.allclose(P, np.eye(3), atol=1e-4))
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'KernelsTests.test_additive_kernel']
    unittest.main()
