#!/usr/bin/env python
import unittest

import numpy as np
import torch
from linear_operator import to_dense
from scipy.special import comb

from epik.kernel import (
    AdditiveKernel,
    ConnectednessKernel,
    ExponentialKernel,
    GeneralProductKernel,
    JengaKernel,
    PairwiseKernel,
    VarianceComponentKernel,
    FactorAnalysisKernel,
    SiteKernelAligner
)
from epik.utils import encode_seqs, get_full_space_one_hot


class KernelsTests(unittest.TestCase):
    def test_additive_kernel(self):
        sl, a = 1, 2
        x = get_full_space_one_hot(sl, a)
        Identity = torch.eye(a**sl)

        # Additive kernel with lambdas1 = 0 should return a constant matrix
        log_lambdas0 = torch.tensor([0.0, -10.0])
        kernel = AdditiveKernel(n_alleles=a, seq_length=sl, log_lambdas0=log_lambdas0)
        cov = kernel.forward(x, x).detach().numpy()
        assert np.allclose(cov, 1, atol=0.01)

        cov = kernel.forward(x, x, diag=True).detach().numpy()
        assert cov.shape == (2,)
        assert np.allclose(cov, 1, atol=0.01)

        # Additive kernel with lambdas0 = 1 should return a purely additive cov
        log_lambdas0 = torch.tensor([-10.0, 0.0])
        kernel = AdditiveKernel(n_alleles=a, seq_length=sl, log_lambdas0=log_lambdas0)
        cov = kernel.forward(x, x).detach().numpy()
        assert np.allclose(cov, np.array([[1, -1], [-1, 1]]), atol=0.01)

        cov = kernel.forward(x, x, diag=True).detach().numpy()
        assert cov.shape == (2,)
        assert np.allclose(cov, 1, atol=0.01)

        # Additive kernel with lambdas = 1 should return the identity
        log_lambdas0 = torch.tensor([0.0, 0])
        kernel = AdditiveKernel(n_alleles=a, seq_length=sl, log_lambdas0=log_lambdas0)
        cov = kernel.forward(x, x).detach().numpy()
        assert np.allclose(cov, (a**sl) * Identity, atol=0.01)

        cov = kernel.forward(x, x, diag=True).detach().numpy()
        assert cov.shape == (2,)
        assert np.allclose(cov, a**sl, atol=0.01)

        sl, a = 2, 2
        x = get_full_space_one_hot(sl, a)

        # Constant kernel
        log_lambdas0 = torch.tensor([0.0, -10])
        kernel = AdditiveKernel(n_alleles=a, seq_length=sl, log_lambdas0=log_lambdas0)
        cov = kernel.forward(x, x).detach().numpy()
        assert np.allclose(cov, 1, atol=0.01)

        cov = kernel.forward(x, x, diag=True).detach().numpy()
        assert cov.shape == (4,)
        assert np.allclose(cov, 1, atol=0.01)

        # Additive kernel
        log_lambdas0 = torch.tensor([-10.0, 0])
        kernel = AdditiveKernel(n_alleles=a, seq_length=sl, log_lambdas0=log_lambdas0)
        cov = kernel.forward(x, x).detach().numpy()
        assert np.allclose(cov[0], [2, 0, 0, -2], atol=0.01)

        cov = kernel.forward(x, x, diag=True).detach().numpy()
        assert cov.shape == (4,)
        assert np.allclose(cov, 2, atol=0.01)

        # Additive kernel with larger variance
        log_lambdas0 = torch.tensor([-10.0, np.log(2)]).to(dtype=torch.float32)
        kernel = AdditiveKernel(n_alleles=a, seq_length=sl, log_lambdas0=log_lambdas0)
        cov = kernel.forward(x, x).detach().numpy()
        assert np.allclose(cov[0], [4, 0, 0, -4], atol=0.01)

        cov = kernel.forward(x, x, diag=True).detach().numpy()
        assert cov.shape == (4,)
        assert np.allclose(cov, 4, atol=0.01)

    def test_pairwise_kernel(self):
        sl, a = 2, 2
        x = get_full_space_one_hot(sl, a)

        # Constant kernel
        log_lambdas0 = torch.tensor([0.0, -10, -10])
        kernel = PairwiseKernel(n_alleles=a, seq_length=sl, log_lambdas0=log_lambdas0)
        cov = kernel.forward(x, x).detach().numpy()
        assert np.allclose(cov, 1, atol=0.01)

        cov = kernel.forward(x, x, diag=True).detach().numpy()
        assert cov.shape == (4,)
        assert np.allclose(cov, 1, atol=0.01)

        # Additive kernel
        log_lambdas0 = torch.tensor([-10.0, 0, -10.0])
        kernel = PairwiseKernel(n_alleles=a, seq_length=sl, log_lambdas0=log_lambdas0)
        cov = kernel.forward(x, x).detach().numpy()
        assert np.allclose(cov[0], [2, 0, 0, -2], atol=0.01)

        # Additive kernel with larger variance
        log_lambdas0 = torch.tensor([-10.0, np.log(2), -10.0]).to(dtype=torch.float32)
        kernel = PairwiseKernel(n_alleles=a, seq_length=sl, log_lambdas0=log_lambdas0)
        cov = kernel.forward(x, x).detach().numpy()
        assert np.allclose(cov[0], [4, 0, 0, -4], atol=0.01)

        # Test in longer sequences
        np.random.seed(0)
        a, sl = 2, 16
        n = sl + 1
        x = np.tril(np.ones((sl + 1, sl)), k=-1)
        x = np.stack([x, 1 - x], axis=2).reshape(n, 2 * sl)
        x = torch.tensor(x, dtype=torch.float32)

        log_lambdas0 = torch.tensor([-30.0, 0, -30.0]).to(dtype=torch.float32)
        kernel = PairwiseKernel(n_alleles=a, seq_length=sl, log_lambdas0=log_lambdas0)
        cov1 = kernel.forward(x, x).detach().numpy()
        assert cov1.shape == (x.shape[0], x.shape[0])
        assert np.allclose(cov1, cov1.T)
        v = np.random.normal(size=cov1.shape[0])
        assert np.dot(v, cov1 @ v) > 0.0

    def test_vc_kernel(self):
        sl, a = 2, 2
        x = get_full_space_one_hot(sl, a)

        # k=0
        log_lambdas0 = torch.tensor([0, -20.0, -20.0], dtype=torch.float32)
        kernel = VarianceComponentKernel(
            n_alleles=a, seq_length=sl, log_lambdas0=log_lambdas0
        )
        cov = kernel._nonkeops_forward(x, x).detach().numpy()
        assert np.allclose(cov, 1/4, atol=1e-4)

        diag = kernel._nonkeops_forward(x, x, diag=True).detach().numpy()
        assert np.allclose(diag, np.diag(cov))

        cov2 = to_dense(kernel._keops_forward(x, x)).detach().numpy()
        assert np.allclose(cov2, cov, atol=1e-4)

        # k=1
        log_lambdas0 = torch.tensor([-10.0, 0.0, -10.0], dtype=torch.float32)
        kernel = VarianceComponentKernel(
            n_alleles=a, seq_length=sl, log_lambdas0=log_lambdas0
        )
        cov = kernel._nonkeops_forward(x, x).detach().numpy()
        k1 = np.array(
            [[1, 0, 0, -1], [0, 1, -1, 0], [0, -1, 1, 0], [-1, 0, 0, 1]],
            dtype=np.float32,
        )
        assert np.allclose(cov, k1 / 2, atol=1e-4)

        diag = kernel._nonkeops_forward(x, x, diag=True).detach().numpy()
        assert np.allclose(diag, np.diag(cov))

        cov2 = to_dense(kernel._keops_forward(x, x)).detach().numpy()
        assert np.allclose(cov2, cov, atol=1e-4)

        # k=2
        log_lambdas0 = torch.tensor([-20.0, -20.0, 0.0], dtype=torch.float32)
        kernel = VarianceComponentKernel(
            n_alleles=a, seq_length=sl, log_lambdas0=log_lambdas0
        )
        cov = kernel._nonkeops_forward(x, x).detach().numpy()
        k2 = np.array(
            [[1, -1, -1, 1], [-1, 1, 1, -1], [-1, 1, 1, -1], [1, -1, -1, 1]],
            dtype=np.float32,
        )
        assert np.allclose(cov, k2 / 4., atol=1e-4)

        cov2 = to_dense(kernel._keops_forward(x, x)).detach().numpy()
        assert np.allclose(cov2, cov)

        # With max k set
        log_lambdas0 = torch.tensor([0.0, -10])
        kernel = VarianceComponentKernel(
            n_alleles=a, seq_length=sl, log_lambdas0=log_lambdas0, max_k=1
        )
        cov = kernel._nonkeops_forward(x, x).detach().numpy()
        assert np.allclose(cov, 1 /4., atol=0.01)

        cov2 = to_dense(kernel._keops_forward(x, x)).detach().numpy()
        assert np.allclose(cov2, cov, atol=0.01)

        # Additive kernel
        log_lambdas0 = torch.tensor([-20.0, 0])
        kernel = VarianceComponentKernel(
            n_alleles=a, seq_length=sl, log_lambdas0=log_lambdas0, max_k=1
        )
        cov = kernel._nonkeops_forward(x, x).detach().numpy()
        assert np.allclose(cov[0], [0.5, 0, 0, -0.5], atol=0.01)

        cov2 = to_dense(kernel._keops_forward(x, x)).detach().numpy()
        assert np.allclose(cov2, cov, atol=0.01)

        # Additive kernel with larger variance
        log_lambdas0 = torch.tensor([-10.0, np.log(2)]).to(dtype=torch.float32)
        kernel = VarianceComponentKernel(
            n_alleles=a, seq_length=sl, log_lambdas0=log_lambdas0, max_k=1
        )
        cov = kernel.forward(x, x).detach().numpy()
        assert np.allclose(cov[0], [1, 0, 0, -1], atol=0.01)

        cov2 = to_dense(kernel._keops_forward(x, x)).detach().numpy()
        assert np.allclose(cov2, cov, atol=0.01)

        # Test in larger spaces
        a, sl, n = 2, 40, 3
        x = np.random.choice([0, 1], size=n * sl)
        x = np.vstack([x, 1 - x]).T.reshape((n, 2 * sl))
        x = torch.tensor(x, dtype=torch.float32)

        log_lambdas0 = torch.tensor([-10.0, 1, -30.0]).to(dtype=torch.float32)
        kernel = VarianceComponentKernel(
            n_alleles=a, seq_length=sl, max_k=2, log_lambdas0=log_lambdas0
        )
        cov = kernel._nonkeops_forward(x, x).detach().numpy()
        cov2 = to_dense(kernel._keops_forward(x, x)).detach().numpy()
        assert np.allclose(cov2, cov, atol=1e-4)

        # Test PSD
        a, sl, n = 4, 8, 1000
        kernel = VarianceComponentKernel(n_alleles=a, seq_length=sl)

        for _ in range(10):
            seqs = ["".join(c) for c in np.random.choice(["A", "C", "G", "T"], size=(n, sl))]
            x = encode_seqs(seqs, alphabet='ACGT')
            K_xx = kernel(x, x)
            for _ in range(10):
                v = torch.rand(n)
                assert torch.dot(v, K_xx @ v) >= 0.0

    def test_exponential_kernel(self):
        sl, a = 2, 2
        theta0 = torch.Tensor(-np.log([2]))
        decay_factor = 1 / 3.0
        kernel = ExponentialKernel(n_alleles=a, seq_length=sl, theta0=theta0)

        x = get_full_space_one_hot(sl, a)
        cov = kernel._nonkeops_forward(x, x).detach().numpy()
        assert np.allclose(cov[0, :], [1, decay_factor, decay_factor, decay_factor**2])

        cov2 = kernel._keops_forward(x, x).to_dense().detach().numpy()
        assert np.allclose(cov, cov2)

        diag = kernel.forward(x, x, diag=True).detach().numpy()
        assert np.allclose(diag, np.diag(cov))

        # Check decay rate
        delta = kernel.get_delta().detach().numpy()
        assert(np.allclose(delta, 1 - decay_factor))
        
        # Check random initialization
        kernel = ExponentialKernel(n_alleles=a, seq_length=sl)
        cov1 = kernel._nonkeops_forward(x, x).detach().numpy()
        cov2 = kernel._keops_forward(x, x).detach().numpy()
        assert(np.allclose(cov1, cov2))

        # Check that it works for theta0 > 0
        theta0 = torch.Tensor(np.log([2]))
        decay_factor = -1 / 3.0
        kernel = ExponentialKernel(n_alleles=a, seq_length=sl, theta0=theta0)
        x = get_full_space_one_hot(sl, a)
        cov = kernel._nonkeops_forward(x, x).detach().numpy()
        assert np.allclose(cov[0, :], [1, decay_factor, decay_factor, decay_factor**2])

    def test_connectedness_kernel(self):
        sl, a = 1, 2
        log_rho = -np.log(2) * torch.ones(sl)
        kernel = ConnectednessKernel(n_alleles=a, seq_length=sl, theta0=log_rho)
        x = get_full_space_one_hot(sl, a)
        cov = kernel._nonkeops_forward(x, x).detach().numpy()
        assert np.allclose(cov[0, :], [1, 1 / 3.0])

        cov2 = kernel._keops_forward(x, x).to_dense().detach().numpy()
        assert np.allclose(cov2, cov)

        diag = kernel.forward(x, x, diag=True).detach().numpy()
        assert np.allclose(diag, np.diag(cov))
        
        # with 2 sites
        sl, a = 2, 2
        log_rho = -np.log(2) * torch.ones(sl)
        kernel = ConnectednessKernel(n_alleles=a, seq_length=sl, theta0=log_rho)
        x = get_full_space_one_hot(sl, a)
        cov = kernel.forward(x, x).detach().numpy()
        assert np.allclose(cov[0, :], [1, 1 / 3.0, 1 / 3.0, 1 / 3.0**2])

        diag = kernel.forward(x, x, diag=True).detach().numpy()
        assert np.allclose(diag, np.diag(cov))

        cov2 = kernel._keops_forward(x, x).to_dense().detach().numpy()
        assert np.allclose(cov, cov)
        
        # With different variance
        kernel = ConnectednessKernel(n_alleles=a, seq_length=sl,
                                     theta0=log_rho,
                                     log_var0=torch.Tensor([-1.]))
        cov1 = kernel._nonkeops_forward(x, x).detach().numpy()
        cov2 = kernel._keops_forward(x, x).detach().numpy()
        assert np.allclose(cov1, cov2)

        # With unequal decay factors
        log_rho = torch.tensor([-np.log(2), -np.log(3)], dtype=torch.float32)
        rho = torch.exp(log_rho)
        decay_factors = (1 - rho) / (1 + rho)

        kernel = ConnectednessKernel(n_alleles=a, seq_length=sl, theta0=log_rho)

        cov = kernel.forward(x, x).detach().numpy()
        expected = np.array(
            [1, decay_factors[0], decay_factors[1], decay_factors[0] * decay_factors[1]]
        )
        assert np.allclose(cov[0, :], expected)

        cov2 = kernel._keops_forward(x, x).to_dense().detach().numpy()
        assert np.allclose(cov2, cov)
        
        # Check decay rates calculation
        delta = kernel.get_delta().detach().numpy()
        expected_delta = 1 - decay_factors
        assert(np.allclose(delta, expected_delta))
        
        # Check random initialization
        kernel = ConnectednessKernel(n_alleles=a, seq_length=sl)
        cov1 = kernel._nonkeops_forward(x, x).detach().numpy()
        cov2 = kernel._keops_forward(x, x).detach().numpy()
        assert(np.allclose(cov1, cov2))
        
        # Check longer sequences
        sl, a = 6, 4
        kernel = ConnectednessKernel(n_alleles=a, seq_length=sl)
        x = get_full_space_one_hot(sl, a)
        cov1 = kernel._nonkeops_forward(x, x).detach().numpy()
        cov2 = kernel._keops_forward(x, x).to_dense().detach().numpy()
        assert np.allclose(cov2, cov1)
        
        # Check longer sequences
        kernel = ConnectednessKernel(n_alleles=a, seq_length=sl,
                                     log_var0=torch.Tensor([-1.]))
        x = get_full_space_one_hot(sl, a)
        cov1 = kernel._nonkeops_forward(x, x).detach().numpy()
        cov2 = kernel._keops_forward(x, x).to_dense().detach().numpy()
        assert np.allclose(cov2, cov1)

    def test_jenga_kernel(self):
        sl, a = 1, 3
        theta0 = torch.tensor(np.log([[0.5, 0.2, 0.6, 0.2]]), dtype=torch.float32)
        kernel = JengaKernel(n_alleles=a, seq_length=sl, theta0=theta0)
        x = get_full_space_one_hot(sl, a)
        rho, eta = 0.5, np.array([4.0, 2 / 3, 4.0])
        allele_factors = np.sqrt(1 + rho * eta)
        a01, a02, a12 = (
            allele_factors[0] * allele_factors[1],
            allele_factors[0] * allele_factors[2],
            allele_factors[1] * allele_factors[2],
        )
        expected = np.array(
            [
                [1, (1 - rho) / a01, (1 - rho) / a02],
                [(1 - rho) / a01, 1, (1 - rho) / a12],
                [(1 - rho) / a02, (1 - rho) / a12, 1],
            ]
        )

        cov = kernel.forward(x, x).detach().numpy()
        assert np.allclose(cov, expected)

        cov2 = kernel._keops_forward(x, x).to_dense().detach().numpy()
        assert np.allclose(cov2, cov)

        # With two sites
        sl, a = 2, 2
        theta0 = torch.tensor(
            np.log([[0.5, 0.2, 0.8], [0.5, 0.5, 0.5]]), dtype=torch.float32
        )
        kernel = JengaKernel(n_alleles=a, seq_length=sl, theta0=theta0)
        x = get_full_space_one_hot(sl, a)
        rho = np.array([0.5, 0.5])
        eta = np.array([[4.0, 0.25], [1.0, 1.0]])
        cov = kernel.forward(x, x).detach().numpy()
        expected = np.array(
            [
                1,
                (1 - rho[0])
                / np.sqrt((1 + rho[0] * eta[0, 0]) * (1 + rho[0] * eta[0, 1])),
                (1 - rho[1])
                / np.sqrt((1 + rho[1] * eta[1, 0]) * (1 + rho[1] * eta[1, 1])),
                np.nan,
            ]
        )
        expected[3] = expected[1] * expected[2]
        assert np.allclose(cov[0, :], expected)

        cov2 = kernel._keops_forward(x, x).to_dense().detach().numpy()
        assert np.allclose(cov2, cov)

        # Check decay rates
        rho = np.expand_dims(rho, 1)
        delta = kernel.get_delta().detach().numpy()
        expected_delta = 1 - np.sqrt((1-rho) / (1 + eta * rho))
        assert(np.allclose(delta, expected_delta))
        
        # Check random initialization
        kernel = JengaKernel(n_alleles=a, seq_length=sl)
        cov1 = kernel._nonkeops_forward(x, x).detach().numpy()
        cov2 = kernel._keops_forward(x, x).detach().numpy()
        assert(np.allclose(cov1, cov2))
        
        # Check longer sequences
        sl, a = 6, 4
        kernel = JengaKernel(n_alleles=a, seq_length=sl)
        x = get_full_space_one_hot(sl, a)
        cov1 = kernel._nonkeops_forward(x, x).detach().numpy()
        cov2 = kernel._keops_forward(x, x).to_dense().detach().numpy()
        assert np.allclose(cov2, cov1)
        
    def test_general_product_kernel(self):
        sl, a = 2, 2
        x = get_full_space_one_hot(sl, a)

        theta0 = torch.full((sl, 1), fill_value=0.0)
        kernel = GeneralProductKernel(a, sl, theta0=theta0)
        K = kernel._nonkeops_forward(x, x).detach().numpy()
        assert np.allclose(K, np.eye(a**sl))

        K = kernel._keops_forward(x, x).to_dense().detach().numpy()
        assert np.allclose(K, np.eye(a**sl))
        
        # Check decay factors
        delta = kernel.get_delta().detach().numpy()
        assert(np.allclose(delta, 1 - np.eye(a)))

        # Check non HOC model
        theta0 = torch.full((sl, 1), fill_value=-1.0)
        kernel = GeneralProductKernel(a, sl, theta0=theta0)
        K1 = kernel._nonkeops_forward(x, x).detach().numpy()
        assert np.any(K1 != np.eye(a**sl))

        K2 = kernel._keops_forward(x, x).to_dense().detach().numpy()
        assert np.allclose(K1, K2)
        
        # Random initialization
        kernel = GeneralProductKernel(a, sl)
        K1 = kernel._nonkeops_forward(x, x).detach().numpy()
        K2 = kernel._keops_forward(x, x).to_dense().detach().numpy()
        assert np.allclose(K1, K2)

        # With larger spaces and random values
        seqs = ["ACGTAGCTAA", "GGGTAGCTAA", "GGGTAGCTCC"]
        x = encode_seqs(seqs, alphabet="ACGT")
        a, sl = 4, 10
        ncomb = int(comb(a, 2))
        theta0 = torch.normal(torch.zeros(size=(sl, ncomb)))
        kernel = GeneralProductKernel(a, sl, theta0=theta0)
        K1 = kernel._nonkeops_forward(x, x).detach().numpy()
        K2 = kernel._keops_forward(x, x).to_dense().detach().numpy()
        assert np.allclose(K1, K2)
        assert np.allclose(np.diag(K1), 1.0)
    
    def test_factor_analysis_kernel(self):
        sl, a, k = 4, 4, 3
        x = get_full_space_one_hot(sl, a)

        kernel = FactorAnalysisKernel(a, sl, k)
        K = kernel.forward(x, x).detach().numpy()
        assert(np.allclose(np.diag(K), 1.0))
        assert(np.allclose(K, K.T))

        # Ensure PSD
        for _ in range(10):
            v = np.random.normal(size=a**sl)
            assert(np.dot(v, K @ v) >= 0.)

        K = kernel.forward(x, x, diag=True).detach().numpy()
        assert np.allclose(K, 1.0)
        
    def test_kernel_aligner(self):
        n_alleles, seq_length = 2, 2
        aligner = SiteKernelAligner(n_alleles=n_alleles, 
                                    seq_length=seq_length)
        
        # Test Exponential transforms
        theta1 = np.array([[-1]])
        
        theta2 = aligner.exponential_to_connectedness(theta1)
        assert(theta2.shape == (2, 1))
        assert(np.allclose(theta2[0], -1))
        assert(np.allclose(aligner.connectedness_to_exponential(theta2), theta1))
        
        theta3 = aligner.exponential_to_jenga(theta1)
        assert(np.allclose(theta3[:, [0]], -1))
        assert(np.allclose(theta3[:, 1:], 0.))
        assert(np.allclose(aligner.jenga_to_exponential(theta3), theta1))
        
        theta4 = aligner.exponential_to_general_product(theta1)
        assert(np.allclose(aligner.general_product_to_exponential(theta4), theta1))
        
        # Test Connectedness transforms
        theta1 = np.array([[-1],
                           [-2]])
        
        theta2 = aligner.connectedness_to_exponential(theta1)
        assert(theta2.shape == (1, 1))
        assert(np.allclose(aligner.log_rho_to_q(theta2[0])[0], 0.61185566))
        
        theta3 = aligner.connectedness_to_jenga(theta1)
        assert(theta3.shape == (seq_length, n_alleles + 1))
        assert(np.allclose(theta3[:, [0]], theta1))
        assert(np.allclose(theta3[:, 1:], 0.))
        assert(np.allclose(aligner.jenga_to_connectedness(theta3), theta1))
        
        theta4 = aligner.connectedness_to_general_product(theta1)
        assert(np.allclose(aligner.general_product_to_connectedness(theta4), theta1))
        
        # Test Jenga transform
        n_alleles, seq_length = 4, 4
        theta1 = np.random.normal(size=(seq_length, n_alleles + 1))
        aligner = SiteKernelAligner(n_alleles=n_alleles, 
                                    seq_length=seq_length)
        
        theta2 = aligner.jenga_to_general_product(theta1)
        theta3 = aligner.general_product_to_jenga(theta2)
        for i in range(seq_length):
            c1 = aligner.jenga_to_corr(theta1[i])
            c3 = aligner.jenga_to_corr(theta3[i])
            assert(np.allclose(c1, c3, atol=1e-2))
        

    # def test_connectedness_site_kernel(self):
    #     sl, a = 2, 2
    #     x = get_full_space_one_hot(sl, a)
    #     theta0 = torch.full((1,), fill_value=np.log(0.5))
    #     rho0 = torch.exp(theta0)[0].item()
    #     r0 = (1 - rho0) / (1 + (a - 1) * rho0)
    #     k0 = np.ones((2, 2))
    #     k1 = np.array([[1, r0], [r0, 1]])
    #     K1 = np.kron(k0, k1)
    #     K2 = np.kron(k1, k0)

    #     # Site 1 kernel
    #     kernel = ConnectednessSiteKernel(a, site=0, theta0=theta0)
    #     K = kernel._nonkeops_forward(x, x).detach().numpy()
    #     assert np.allclose(K, K1)

    #     K = kernel._keops_forward(x, x).to_dense().detach().numpy()
    #     assert np.allclose(K, K1)

    #     K_diag = kernel._nonkeops_forward(x, x, diag=True).detach().numpy()
    #     assert K_diag.shape == (x.shape[0],)
    #     assert np.allclose(K_diag, 1.0)

    #     # Site 2 kernel
    #     kernel = ConnectednessSiteKernel(a, site=1, theta0=theta0)
    #     K = kernel._nonkeops_forward(x, x).detach().numpy()
    #     assert np.allclose(K, K2)

    #     K = kernel._keops_forward(x, x).to_dense().detach().numpy()
    #     assert np.allclose(K, K2)

    #     K_diag = kernel._nonkeops_forward(x, x, diag=True).detach().numpy()
    #     assert K_diag.shape == (x.shape[0],)
    #     assert np.allclose(K_diag, 1.0)

    # def test_jenga_site_kernel(self):
    #     sl, a = 2, 3
    #     x = get_full_space_one_hot(sl, a)
    #     theta0 = torch.Tensor(-np.log([2, 2, 4, 4]))
    #     r1 = 0.5 / (np.sqrt(2.5) * np.sqrt(1.5))
    #     r2 = 1 / 5
    #     k0 = np.ones((3, 3))
    #     k1 = np.array([[1, r1, r1], [r1, 1, r2], [r1, r2, 1]])
    #     K1 = np.kron(k0, k1)
    #     K2 = np.kron(k1, k0)

    #     # Site 1 kernel
    #     kernel = JengaSiteKernel(a, site=0, theta0=theta0)
    #     K = kernel._nonkeops_forward(x, x).detach().numpy()
    #     assert np.allclose(K, K1)

    #     K = kernel._keops_forward(x, x).to_dense().detach().numpy()
    #     assert np.allclose(K, K1)

    #     K_diag = kernel._nonkeops_forward(x, x, diag=True).detach().numpy()
    #     assert K_diag.shape == (x.shape[0],)
    #     assert np.allclose(K_diag, 1.0)

    #     # Site 2 kernel
    #     kernel = JengaSiteKernel(a, site=1, theta0=theta0)
    #     K = kernel._nonkeops_forward(x, x).detach().numpy()
    #     assert np.allclose(K, K2)

    #     K = kernel._keops_forward(x, x).to_dense().detach().numpy()
    #     assert np.allclose(K, K2)

    #     K_diag = kernel._nonkeops_forward(x, x, diag=True).detach().numpy()
    #     assert K_diag.shape == (x.shape[0],)
    #     assert np.allclose(K_diag, 1.0)

    # def test_general_site_kernel(self):
    #     sl, a = 2, 2
    #     x = get_full_space_one_hot(sl, a)
    #     K1 = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
    #     K2 = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]])

    #     theta0 = torch.full((1,), fill_value=0.0)

    #     # Site 1 kernel
    #     kernel = GeneralSiteKernel(a, site=0, theta0=theta0)
    #     K = kernel._nonkeops_forward(x, x).detach().numpy()
    #     assert np.allclose(K, K1)

    #     K = kernel._keops_forward(x, x).to_dense().detach().numpy()
    #     assert np.allclose(K, K1)

    #     K_diag = kernel._nonkeops_forward(x, x, diag=True).detach().numpy()
    #     assert K_diag.shape == (x.shape[0],)
    #     assert np.allclose(K_diag, 1.)

    #     # Site 2 kernel
    #     kernel = GeneralSiteKernel(a, site=1, theta0=theta0)
    #     K = kernel._nonkeops_forward(x, x).detach().numpy()
    #     assert np.allclose(K, K2)

    #     K = kernel._keops_forward(x, x).to_dense().detach().numpy()
    #     assert np.allclose(K, K2)

    #     K_diag = kernel._nonkeops_forward(x, x, diag=True).detach().numpy()
    #     assert K_diag.shape == (x.shape[0],)
    #     assert np.allclose(K_diag, 1.0)

    # def test_rho_pi_kernel(self):
    #     sl, a = 1, 2
    #     logit_rho0 = torch.tensor([[0.]])
    #     log_p0 = torch.tensor(np.log([[0.2, 0.8]]), dtype=torch.float32)
    #     kernel = RhoPiKernel(n_alleles=a, seq_length=sl,
    #                          logit_rho0=logit_rho0, log_p0=log_p0)
    #     x = get_full_space_one_hot(sl, a)
    #     cov = kernel.forward(x, x).detach().numpy()
    #     expected = np.array([[3, 0.5],
    #                          [0.5, 1.125]])
    #     assert(np.allclose(cov, expected))

    #     diag = kernel.forward(x, x, diag=True).detach().numpy()
    #     assert(np.allclose(diag, np.diag(cov)))

    #     cov2 = to_dense(kernel._keops_forward(x, x)).detach().numpy()
    #     assert(np.allclose(cov2, cov))

    #     sl, a = 2, 2
    #     logit_rho0 = torch.tensor([[0.],
    #                                [0.]], dtype=torch.float32)
    #     log_p0 = torch.tensor(np.log([[0.2, 0.8],
    #                                   [0.5, 0.5]]), dtype=torch.float32)
    #     kernel = RhoPiKernel(n_alleles=a, seq_length=sl,
    #                          logit_rho0=logit_rho0, log_p0=log_p0)
    #     x = get_full_space_one_hot(sl, a)
    #     rho = np.array([0.5, 0.5])
    #     eta = np.array([[4., 0.25],
    #                     [1., 1.  ]])
    #     cov = kernel.forward(x, x).detach().numpy()
    #     expected = np.array([(1 + rho[0] * eta[0, 0]) * (1 + rho[1] * eta[1, 0]),
    #                          (1 - rho[0])             * (1 + rho[1] * eta[1, 0]),
    #                          (1 + rho[0] * eta[0, 0]) * (1 - rho[1]),
    #                          (1 - rho[0])             * (1 - rho[1])])
    #     assert(np.allclose(cov[0, :], expected))

    #     diag = kernel.forward(x, x, diag=True).detach().numpy()
    #     assert(np.allclose(diag, np.diag(cov)))

    #     cov2 = to_dense(kernel._keops_forward(x, x)).detach().numpy()
    #     assert(np.allclose(cov2, cov))

    # def test_heteroskedastic_kernel(self):
    #     sl, a = 1, 2
    #     x = get_full_space_one_hot(sl, a)

    #     logit_rho0 = torch.tensor([[0.0]])
    #     kernel = ConnectednessKernel(n_alleles=a, seq_length=sl, logit_rho0=logit_rho0)
    #     cov1 = kernel.forward(x, x)
    #     assert cov1[0, 0] == 1.5
    #     assert cov1[0, 1] == 0.5

    #     kernel = AdditiveHeteroskedasticKernel(kernel)
    #     cov2 = kernel.forward(x, x)
    #     assert cov2[0, 0] < 1.5
    #     assert cov2[0, 1] < 0.5

    # def test_keops(self):
    #     from pykeops.torch import LazyTensor

    #     # Example inputs x1 and x2, shapes: (batch_size, n, d)
    #     x1 = torch.randn(10, 100, 10)  # Shape: (batch_size, n, d)
    #     x2 = torch.randn(10, 100, 10)  # Shape: (batch_size, n, d)

    #     # Step 1: Convert x1 and x2 into LazyTensor objects
    #     x1_lazy = LazyTensor(x1[:, None, :, :])  # Shape: (batch_size, n, 1, d)
    #     x2_lazy = LazyTensor(x2[:, :, None, :])  # Shape: (batch_size, n, d, 1)

    #     # Step 2: Compute the element-wise product over the last axis
    #     elementwise_product = x1_lazy * x2_lazy  # Shape: (batch_size, n, d, d)

    #     # Step 3: Perform sum reduction over axis `a` (axis=-2 in this case)
    #     summed_result = elementwise_product.sum(-1)  # Shape: (batch_size, n, d)
    #     print(summed_result.shape)

    #     # Step 4: Perform product reduction over axis `l` (axis=-1 in this case)
    #     final_result = summed_result.sum(-3)  # Shape: (batch_size, n)

    #     print(final_result.shape)


if __name__ == "__main__":
    import sys

    sys.argv = ["", "KernelsTests"]
    unittest.main()
