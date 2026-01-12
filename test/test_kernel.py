#!/usr/bin/env python
import unittest

import numpy as np
import torch

from torch.nn import Parameter
from scipy.special import comb

from epik.kernel import (
    AdditiveKernel,
    ConnectednessKernel,
    GeometricKernel,
    GeneralProductKernel,
    JengaKernel,
    PairwiseKernel,
    VarianceComponentKernel,
    MahalanobisRBFKernel,
    FactorAnalysisKernel,
    SiteKernelAligner,
)
from epik.utils import encode_seqs, get_full_space_one_hot


class KernelsTests(unittest.TestCase):
    def setUp(self):
        self.kernels = [
            AdditiveKernel,
            PairwiseKernel,
            VarianceComponentKernel,
            #    GeometricKernel,
            #    ConnectednessKernel, JengaKernel, GeneralProductKernel,
        ]

        self.alphabet = list('AB')
        self.config = {'alphabet': self.alphabet, 'seq_length': 2}
        self.alpha = 2
        self.l = 2
        self.n = 4
        self.x = get_full_space_one_hot(self.l, self.alpha)

        return super().setUp()

    def test_kernel_configurations(self):
        configs = [
            {"alphabet_type": "dna", "seq_length": 4},
            {"alphabet": list("ABC"), "seq_length": 5},
            {"alphabet_list": [list("ABC")] * 3},
        ]

        for kernel in self.kernels:
            for kwargs in configs:
                kernel(**kwargs)

    def test_kernel_erroneous_configurations(self):
        configs = [
            {"alphabet_type": "dna", "alphabet": list("AC")},
            {"alphabet": list("ABC"), "seq_length": 5, "alphabet_type": "dna"},
            {"alphabet_type": "rna"},
            {"alphabet_type": "xxx", "seq_length": 4},
        ]

        for kernel in self.kernels:
            for kwargs in configs:
                try:
                    kernel(**kwargs)
                except ValueError:
                    pass
    
    def test_select_site(self): 
        kernel = GeometricKernel(**self.config)
        assert np.all(kernel.starts == [0, 2])
        assert np.all(kernel.ends == [2, 4])

        x0 = kernel.select_site(self.x, site=0)
        x1 = kernel.select_site(self.x, site=1)
        assert np.allclose(x0, self.x[:, :2])
        assert np.allclose(x1, self.x[:, 2:])

    def test_additive_kernel(self):
        Identity = torch.eye(self.n)
        kernel = AdditiveKernel(**self.config)

        log_lambdas = [[0.0, -10.0], [-10.0, 0.0]]
        covs = [
            1,
            np.array([[2, 0, 0, -2], [0, 2, -2, 0], [0, -2, 2, 0], [-2, 0, 0, 2]]),
            self.n * Identity,
        ]
        s2s = [1, 2, 4]
        for log_lambda, cov, s2 in zip(log_lambdas, covs, s2s):
            kernel.log_lambdas = Parameter(torch.tensor(log_lambda))
            k = kernel.forward(self.x, self.x).detach().numpy()
            diag = kernel.forward(self.x, self.x, diag=True).detach().numpy()
            assert np.allclose(k, cov, atol=0.01)
            assert diag.shape == (self.n,)
            assert np.allclose(diag, s2, atol=0.01)
        
        # Test in longer sequences
        sl = 16
        n = sl + 1
        x = np.tril(np.ones((sl + 1, sl)), k=-1)
        x = np.stack([x, 1 - x], axis=2).reshape(n, 2 * sl)
        x = torch.tensor(x, dtype=torch.float32)

        log_lambdas0 = torch.tensor([-30.0, 0]).to(dtype=torch.float32)
        kernel = AdditiveKernel(
            alphabet=self.alphabet, seq_length=sl, log_lambdas0=log_lambdas0
        )
        k = kernel.forward(x, x).detach().numpy()
        v = np.random.normal(size=k.shape[0])
        assert k.shape == (x.shape[0], x.shape[0])
        assert np.allclose(k, k.T)
        assert np.dot(v, k @ v) > 0.0

    def test_pairwise_kernel(self):
        kernel = PairwiseKernel(**self.config)

        log_lambdas = [[0.0, -10, -10], [-10, 0.0, -10], [-10.0, np.log(2), -10.0]]
        covs = [1, [2, 0, 0, -2], [4, 0, 0, -4]]
        s2s = [1, 2, 4]

        for log_lambda, cov, s2 in zip(log_lambdas, covs, s2s):
            log_lambda = torch.tensor(log_lambda).to(dtype=torch.float32)
            kernel.log_lambdas = Parameter(log_lambda)
            
            # With GPyTorch
            k = kernel._nonkeops_forward(self.x, self.x).detach().numpy()[0]
            diag = kernel._nonkeops_forward(self.x, self.x, diag=True).detach().numpy()
            assert np.allclose(k, cov, atol=0.01)
            assert diag.shape == (self.n,)
            assert np.allclose(diag, s2, atol=0.01)

            # With KeOps
            k = kernel._keops_forward(self.x, self.x).detach().numpy()[0]
            assert np.allclose(k, cov, atol=0.01)

        # Test in longer sequences
        sl = 16
        n = sl + 1
        x = np.tril(np.ones((sl + 1, sl)), k=-1)
        x = np.stack([x, 1 - x], axis=2).reshape(n, 2 * sl)
        x = torch.tensor(x, dtype=torch.float32)

        log_lambdas0 = torch.tensor([-30.0, 0, -30.0]).to(dtype=torch.float32)
        kernel = PairwiseKernel(
            alphabet=self.alphabet, seq_length=sl, log_lambdas0=log_lambdas0
        )
        cov1 = kernel.forward(x, x).detach().numpy()
        v = np.random.normal(size=cov1.shape[0])
        assert cov1.shape == (x.shape[0], x.shape[0])
        assert np.allclose(cov1, cov1.T)
        assert np.dot(v, cov1 @ v) > 0.0

    def test_vc_kernel(self):
        kernel = VarianceComponentKernel(**self.config)

        log_lambdas = [[0, -20.0, -20.0], [-10.0, 0.0, -10.0], [-20.0, -20.0, 0.0]]
        covs = [1, [2, 0, 0, -2], [1, -1, -1, 1]]
        s2s = [1, 2, 1]

        for log_lambda, cov, s2 in zip(log_lambdas, covs, s2s):
            log_lambda = torch.tensor(log_lambda).to(dtype=torch.float32)
            kernel.log_lambdas = Parameter(log_lambda)

            # With GPyTorch
            k = kernel._nonkeops_forward(self.x, self.x).detach().numpy()[0]
            diag = kernel._nonkeops_forward(self.x, self.x, diag=True).detach().numpy()
            assert np.allclose(k, cov, atol=0.01)
            assert diag.shape == (self.n,)
            assert np.allclose(diag, s2, atol=0.01)

            # With KeOps
            k = kernel._keops_forward(self.x, self.x).detach().numpy()[0]
            assert np.allclose(k, cov, atol=0.01)

    def test_truncated_vc_kernel(self):
        alphabet = list('ACGT')
        sl, n = 2, 100
        seqs = ["".join(c) for c in np.random.choice(alphabet, size=(n, sl))]
        x = encode_seqs(seqs, alphabet=alphabet)
        v = np.random.normal(size=n)

        # Additive kernel
        log_lambdas0 = torch.tensor([-20.0, 0.0], dtype=torch.float32)
        kernel1 = VarianceComponentKernel(
            alphabet=alphabet, seq_length=sl, max_k=1, log_lambdas0=log_lambdas0
        )
        kernel2 = AdditiveKernel(
            alphabet=alphabet, seq_length=sl, log_lambdas0=log_lambdas0
        )
        k1 = kernel1(x, x).detach().numpy()
        k2 = kernel2(x, x).detach().numpy()
        assert np.allclose(k1, k2, atol=1e-4)
        assert np.dot(v, k1 @ v) >= 0.0

        # Pairwise kernel
        log_lambdas0 = torch.tensor([-20.0, -20.0, 0.0], dtype=torch.float32)
        kernel1 = VarianceComponentKernel(
            alphabet=alphabet, seq_length=sl, max_k=2, log_lambdas0=log_lambdas0
        )
        kernel2 = PairwiseKernel(
            alphabet=alphabet, seq_length=sl, log_lambdas0=log_lambdas0
        )
        k1 = kernel1(x, x).detach().numpy()
        k2 = kernel2(x, x).detach().numpy()
        assert np.allclose(k1, k2, atol=1e-4)
        assert np.dot(v, k1 @ v) >= 0.0

    def test_geometric_kernel(self):
        config = self.config.copy()
        config.update({"theta0": torch.Tensor(-np.log([2]))})
        kernel = GeometricKernel(**config)
        corr1d = 1 / 3.0
        corrs = [1, corr1d, corr1d, corr1d**2]

        # Check decay factor
        delta = kernel.get_delta().detach().numpy()
        assert np.allclose(1 - delta, corr1d)

        # Check site kernels
        ks = kernel.get_site_kernels()
        logks = kernel.get_site_log_kernels()
        for k, logk in zip(ks, logks):
            assert np.allclose(k.detach(), torch.exp(logk.detach()))

        # Check kernel calculation
        cov = kernel._nonkeops_forward(self.x, self.x).detach().numpy()
        assert np.allclose(cov[0, :], corrs)

        cov2 = kernel._keops_forward(self.x, self.x).to_dense().detach().numpy()
        assert np.allclose(cov, cov2)

        diag = kernel.forward(self.x, self.x, diag=True).detach().numpy()
        assert np.allclose(diag, np.diag(cov))

        # Check random initialization
        kernel = GeometricKernel(**self.config)
        cov1 = kernel._nonkeops_forward(self.x, self.x).detach().numpy()
        cov2 = kernel._keops_forward(self.x, self.x).detach().numpy()
        assert np.allclose(cov1, cov2)

        # Check that it works for theta0 > 0
        config["theta0"] = torch.Tensor(np.log([2]))
        kernel = GeometricKernel(**config)
        corr1d = -1 / 3.0

        # Check decay factor
        delta = kernel.get_delta().detach().numpy()
        assert np.allclose(1 - delta, corr1d)

        corrs = [1, corr1d, corr1d, corr1d**2]
        cov = kernel._nonkeops_forward(self.x, self.x).detach().numpy()
        assert np.allclose(cov[0, :], corrs)
        
        cov2 = kernel._keops_forward(self.x, self.x).detach().numpy()
        assert np.allclose(cov2, cov)

    def test_connectedness_kernel(self):
        config = self.config.copy()
        config.update({"theta0": torch.Tensor([-np.log(2), 0.])})
        kernel = ConnectednessKernel(**config)
        corr1d = [1 / 3.0, 0]
        corrs = [1, corr1d[0], corr1d[1],corr1d[0] * corr1d[1]]

        # Check decay factor
        delta = kernel.get_delta().detach()
        assert np.allclose(1 - delta, corr1d)

        # Check site kernels
        ks = kernel.get_site_kernels()
        logks = kernel.get_site_log_kernels()
        for k, logk in zip(ks, logks):
            assert np.allclose(k.detach(), torch.exp(logk.detach()))

        # Check kernel calculation
        cov = kernel._nonkeops_forward(self.x, self.x).detach().numpy()
        assert np.allclose(cov[0, :], corrs)

        cov2 = kernel._keops_forward(self.x, self.x).to_dense().detach().numpy()
        assert np.allclose(cov, cov2)

        diag = kernel.forward(self.x, self.x, diag=True).detach().numpy()
        assert np.allclose(diag, np.diag(cov))

        # Check random initialization
        kernel = ConnectednessKernel(**self.config)
        cov1 = kernel._nonkeops_forward(self.x, self.x).detach().numpy()
        cov2 = kernel._keops_forward(self.x, self.x).detach().numpy()
        assert np.allclose(cov1, cov2)

        # Check that it works for theta0 > 0
        config["theta0"] = torch.Tensor(np.log([2, 1]))
        kernel = ConnectednessKernel(**config)
        corr1d = [-1 / 3.0, 0]

        delta = kernel.get_delta().detach().numpy()
        assert np.allclose(1 - delta, corr1d)

        corrs = [1, corr1d[0], corr1d[1], corr1d[0] * corr1d[1]]
        cov = kernel._nonkeops_forward(self.x, self.x).detach().numpy()
        assert np.allclose(cov[0, :], corrs)
        
        cov2 = kernel._keops_forward(self.x, self.x).detach().numpy()
        assert np.allclose(cov2, cov)
    
    def test_jenga_kernel(self):
        config = self.config.copy()
        config["log_mu0"] = torch.Tensor([-np.log(2), 0.0])
        config["log_pi0"] = [torch.Tensor([0.0, 0.0]), torch.Tensor([0.0, 1.0])]
        kernel = JengaKernel(**config)
        corr1d = [1 / 3.0, 0]
        corrs = [1, corr1d[0], corr1d[1],corr1d[0] * corr1d[1]]

        # Check decay factor
        delta = kernel.get_delta()#.detach()
        k1 = 1 - delta[0].detach()[0, 1]
        k2 = 1 - delta[1].detach()[0, 1]
        assert np.allclose(k1, corr1d[0])
        assert np.allclose(k2, corr1d[1])

        # Check site kernels
        ks = kernel.get_site_kernels()
        logks = kernel.get_site_log_kernels()
        for k, logk in zip(ks, logks):
            assert np.allclose(k.detach(), torch.exp(logk.detach()))

        # Check kernel calculation
        cov = kernel._nonkeops_forward(self.x, self.x).detach().numpy()
        assert np.allclose(cov[0, :], corrs)

        cov2 = kernel._keops_forward(self.x, self.x).to_dense().detach().numpy()
        assert np.allclose(cov, cov2)

        diag = kernel.forward(self.x, self.x, diag=True).detach().numpy()
        assert np.allclose(diag, np.diag(cov))

        # Check random initialization
        kernel = JengaKernel(**self.config)
        cov1 = kernel._nonkeops_forward(self.x, self.x).detach().numpy()
        cov2 = kernel._keops_forward(self.x, self.x).detach().numpy()
        assert np.allclose(cov1, cov2)

        # Check that it works for theta0 > 0
        config["log_mu0"] = torch.Tensor([np.log(2), 0.0])
        config["log_pi0"] = [torch.Tensor([0.0, 0.0]), torch.Tensor([0.0, 0.0])]
        kernel = JengaKernel(**config)
        corr1d = [-1 / 3.0, 0]

        delta = kernel.get_delta()
        k1 = 1 - delta[0].detach()[0, 1]
        k2 = 1 - delta[1].detach()[0, 1]
        assert np.allclose(k1, corr1d[0])
        assert np.allclose(k2, corr1d[1])

        corrs = [1, corr1d[0], corr1d[1], corr1d[0] * corr1d[1]]
        cov = kernel._nonkeops_forward(self.x, self.x).detach().numpy()
        assert np.allclose(cov[0, :], corrs)
        
        cov2 = kernel._keops_forward(self.x, self.x).detach().numpy()
        assert np.allclose(cov2, cov)

        # With three alleles
        alphabet = list('ACB')
        seq_length = 1
        log_mu = torch.log(torch.Tensor([0.5]))
        log_pi = [torch.log(torch.Tensor([0.2, 0.6, 0.2]))]

        kernel = JengaKernel(
            alphabet=alphabet, seq_length=seq_length, log_mu0=log_mu, log_pi0=log_pi
        )
        x = get_full_space_one_hot(seq_length, len(alphabet))
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

        # Check longer sequences
        alphabet = list("ACBD")
        sl = 6
        kernel = JengaKernel(alphabet=alphabet, seq_length=sl)
        x = get_full_space_one_hot(sl, len(alphabet))
        cov1 = kernel._nonkeops_forward(x, x).detach().numpy()
        cov2 = kernel._keops_forward(x, x).to_dense().detach().numpy()
        assert np.allclose(cov2, cov1)

    def test_general_product_kernel(self):
        config = self.config.copy()
        config['theta0'] = [torch.full((1,), fill_value=0.0)] * 2
        kernel = GeneralProductKernel(**config)
        K_exp = np.eye(4)

        # Check decay factors
        for delta in kernel.get_delta():
            assert np.allclose(delta.detach().numpy(), 1 - np.eye(2))

        # Check site kernels
        ks = kernel.get_site_kernels()
        logks = kernel.get_site_log_kernels()
        for k, logk in zip(ks, logks):
            assert np.allclose(k.detach(), torch.exp(logk.detach()))

        K = kernel._nonkeops_forward(self.x, self.x).detach().numpy()
        assert np.allclose(K, K_exp)

        K = kernel._keops_forward(self.x, self.x).to_dense().detach().numpy()
        assert np.allclose(K, K_exp)

        diag = kernel.forward(self.x, self.x, diag=True).detach().numpy()
        assert np.allclose(diag, np.diag(K))

        # Random initialization
        kernel = GeneralProductKernel(**config)
        K1 = kernel._nonkeops_forward(self.x, self.x).detach().numpy()
        K2 = kernel._keops_forward(self.x, self.x).to_dense().detach().numpy()
        assert np.allclose(K1, K2)

        # With larger spaces
        seqs = ["ACGTAGCTAA", "GGGTAGCTAA", "GGGTAGCTCC"]
        x = encode_seqs(seqs, alphabet="ACGT")
        kernel = GeneralProductKernel(alphabet="ACGT", seq_length=10)
        K1 = kernel._nonkeops_forward(x, x).detach().numpy()
        K2 = kernel._keops_forward(x, x).to_dense().detach().numpy()
        assert np.allclose(K1, K2)
        assert np.allclose(np.diag(K1), 1.0)

    def test_linear_embedding_kernel(self):
        sl, a = 4, 4
        x = get_full_space_one_hot(sl, a)

        kernel = MahalanobisRBFKernel(a, sl)
        K = kernel.forward(x, x).detach().numpy()
        assert np.allclose(np.diag(K), 1.0, atol=1e-3)
        assert np.allclose(K, K.T, atol=1e-4)

        # Ensure PSD
        for _ in range(10):
            v = np.random.normal(size=a**sl)
            assert np.dot(v, K @ v) >= 0.0

        K = kernel.forward(x, x, diag=True).detach().numpy()
        assert np.allclose(K, 1.0)

        # Ensure diagonal is properly computed
        x1, x2 = x[:10, :], x[10:20, :]
        K = kernel.forward(x1, x2).detach().numpy()
        diag = kernel.forward(x1, x2, diag=True).detach().numpy()
        assert np.allclose(diag, np.diag(K))

        corr1ds = kernel.get_M_corr1ds()

    def test_factor_analysis_kernel(self):
        sl, a = 4, 4
        x = get_full_space_one_hot(sl, a)
        kernel = FactorAnalysisKernel(a, sl, ndim=3)
        A = kernel.get_A()
        assert A.shape[1] == 3

        K = kernel.forward(x, x).detach().numpy()
        assert np.allclose(np.diag(K), 1.0, atol=1e-3)
        assert np.allclose(K, K.T, atol=1e-4)

        # Ensure PSD
        for _ in range(10):
            v = np.random.normal(size=a**sl)
            assert np.dot(v, K @ v) >= 0.0

        K = kernel.forward(x, x, diag=True).detach().numpy()
        assert np.allclose(K, 1.0)

        # Ensure diagonal is properly computed
        x1, x2 = x[:10, :], x[10:20, :]
        K = kernel.forward(x1, x2).detach().numpy()
        diag = kernel.forward(x1, x2, diag=True).detach().numpy()
        assert np.allclose(diag, np.diag(K))

    def test_kernel_aligner(self):
        n_alleles, seq_length = 2, 2
        aligner = SiteKernelAligner(n_alleles=n_alleles, seq_length=seq_length)

        # Test Exponential transforms
        theta1 = np.array([[-1]])

        theta2 = aligner.geometric_to_connectedness(theta1)
        assert theta2.shape == (2, 1)
        assert np.allclose(theta2[0], -1)
        assert np.allclose(aligner.connectedness_to_geometric(theta2), theta1)

        theta3 = aligner.geometric_to_jenga(theta1)
        assert np.allclose(theta3[:, [0]], -1)
        assert np.allclose(theta3[:, 1:], 0.0)
        assert np.allclose(aligner.jenga_to_geometric(theta3), theta1)

        theta4 = aligner.geometric_to_general_product(theta1)
        assert np.allclose(aligner.general_product_to_geometric(theta4), theta1)

        # Test Connectedness transforms
        theta1 = np.array([[-1], [-2]])

        theta2 = aligner.connectedness_to_geometric(theta1)
        assert theta2.shape == (1, 1)
        assert np.allclose(aligner.log_rho_to_q(theta2[0])[0], 0.61185566)

        theta3 = aligner.connectedness_to_jenga(theta1)
        assert theta3.shape == (seq_length, n_alleles + 1)
        assert np.allclose(theta3[:, [0]], theta1)
        assert np.allclose(theta3[:, 1:], 0.0)
        assert np.allclose(aligner.jenga_to_connectedness(theta3), theta1)

        theta4 = aligner.connectedness_to_general_product(theta1)
        assert np.allclose(aligner.general_product_to_connectedness(theta4), theta1)

        # Test Jenga transform
        n_alleles, seq_length = 4, 4
        theta1 = np.random.normal(size=(seq_length, n_alleles + 1))
        aligner = SiteKernelAligner(n_alleles=n_alleles, seq_length=seq_length)

        theta2 = aligner.jenga_to_general_product(theta1)
        theta3 = aligner.general_product_to_jenga(theta2)
        for i in range(seq_length):
            c1 = aligner.jenga_to_corr(theta1[i])
            c3 = aligner.jenga_to_corr(theta3[i])
            assert np.allclose(c1, c3, atol=1e-2)

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
