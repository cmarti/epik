#!/usr/bin/env python
import unittest

import numpy as np
import torch

from tempfile import NamedTemporaryFile
from torch.nn import Parameter
from scipy.special import comb

from epik.kernel import (
    AdditiveKernel,
    ConnectednessKernel,
    GeometricKernel,
    GeneralProductKernel,
    GeneralProductKernel2,
    JengaKernel,
    PairwiseKernel,
    VarianceComponentKernel,
    MahalanobisRBFKernel,
    FactorAnalysisKernel,
    ConnectednessFactorAnalysisKernel,
    DiploidKernel,
    SiteKernelAligner,
)
from epik.utils import encode_seqs, get_full_space_one_hot


class KernelsTests(unittest.TestCase):
    def setUp(self):
        self.kernels = [
            AdditiveKernel,
            PairwiseKernel,
            VarianceComponentKernel,
            GeometricKernel,
            ConnectednessKernel,
            JengaKernel,
            GeneralProductKernel,
            MahalanobisRBFKernel,
            FactorAnalysisKernel,
        ]

        self.alphabet = list("AB")
        self.config = {"alphabet": self.alphabet, "seq_length": 2}
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

    def test_load_save(self):
        for kernel in self.kernels:
            kernel = kernel(**self.config)

            with NamedTemporaryFile("w") as fhand:
                kernel.save(fhand.name)
                kernel.load(fhand.name)

    def test_select_site(self):
        kernel = GeometricKernel(**self.config)
        assert np.all(kernel.starts == [0, 2])
        assert np.all(kernel.ends == [2, 4])

        x0 = kernel.select_site(self.x, site=0)
        x1 = kernel.select_site(self.x, site=1)
        assert np.allclose(x0, self.x[:, :2])
        assert np.allclose(x1, self.x[:, 2:])

    def test_select_allele(self):
        kernel = GeometricKernel(**self.config)
        assert kernel.alleles_idx == {"A": [0, 2], "B": [1, 3]}

        x_A = kernel.select_allele(self.x, allele="A")
        x_B = kernel.select_allele(self.x, allele="B")
        assert np.allclose(x_A, self.x[:, [0, 2]])
        assert np.allclose(x_B, self.x[:, [1, 3]])

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
        alphabet = list("ACGT")
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
        config.update({"log_mu0": torch.Tensor(-np.log([2]))})
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

        # Check that it works for log_mu0 > 0
        config["log_mu0"] = torch.Tensor(np.log([2]))
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
        config.update({"log_mu0": torch.Tensor([-np.log(2), 0.0])})
        kernel = ConnectednessKernel(**config)
        corr1d = [1 / 3.0, 0]
        corrs = [1, corr1d[0], corr1d[1], corr1d[0] * corr1d[1]]

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

        # Check that it works for log_mu0 > 0
        config["log_mu0"] = torch.Tensor(np.log([2, 1]))
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
        corrs = [1, corr1d[0], corr1d[1], corr1d[0] * corr1d[1]]

        # Check decay factor
        delta = kernel.get_delta()  # .detach()
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
        alphabet = list("ACB")
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
        config["theta0"] = [torch.full((1,), fill_value=0.0)] * 2
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
        kernel = GeneralProductKernel(alphabet=list("ACGT"), seq_length=10)
        K1 = kernel._nonkeops_forward(x, x).detach().numpy()
        K2 = kernel._keops_forward(x, x).to_dense().detach().numpy()
        assert np.allclose(K1, K2)
        assert np.allclose(np.diag(K1), 1.0)

    def test_general_product_kernel2(self):
        config = self.config.copy()
        config["theta0"] = [torch.full((1,), fill_value=0.0)] * 2
        kernel = GeneralProductKernel2(**config)

        K1 = kernel._nonkeops_forward(self.x, self.x).detach().numpy()
        K2 = kernel._keops_forward(self.x, self.x).to_dense().detach().numpy()
        assert np.allclose(K1, K2)

    def test_general_product_kernel_memory_log_vs_nonlog(self):
        torch.manual_seed(0)
        np.random.seed(0)

        alphabet = list("ACGT")
        seq_length = 12
        n = 1024
        seqs = ["".join(s) for s in np.random.choice(alphabet, size=(n, seq_length))]

        def run_once(use_log_space: bool, device: str):
            x = encode_seqs(seqs, alphabet=alphabet).to(
                device=device, dtype=torch.float32
            )
            kernel = GeneralProductKernel(
                alphabet=alphabet,
                seq_length=seq_length,
                use_keops=False,
                use_log_space=use_log_space,
            ).to(device=device)
            kernel.zero_grad(set_to_none=True)

            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                K = kernel._nonkeops_forward(x, x)
                loss = K.square().mean()
                loss.backward()
                torch.cuda.synchronize()
                peak_alloc_mb = torch.cuda.max_memory_allocated() / 2**20
                peak_reserved_mb = torch.cuda.max_memory_reserved() / 2**20
            else:
                with torch.autograd.profiler.profile(profile_memory=True) as prof:
                    K = kernel._nonkeops_forward(x, x)
                    loss = K.square().mean()
                    loss.backward()
                cpu_mem_bytes = sum(
                    max(0, evt.self_cpu_memory_usage) for evt in prof.function_events
                )
                peak_alloc_mb = cpu_mem_bytes / 2**20
                peak_reserved_mb = peak_alloc_mb

            has_grad = all(
                (p.grad is not None) for p in kernel.parameters() if p.requires_grad
            )

            del K, loss, kernel
            if device == "cuda":
                torch.cuda.empty_cache()
            return peak_alloc_mb, peak_reserved_mb, has_grad

        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            log_alloc, log_reserved, log_has_grad = run_once(
                use_log_space=True, device=device
            )
            nonlog_alloc, nonlog_reserved, nonlog_has_grad = run_once(
                use_log_space=False, device=device
            )

            assert log_alloc > 0
            assert nonlog_alloc > 0
            assert log_reserved > 0
            assert nonlog_reserved > 0
            assert log_has_grad
            assert nonlog_has_grad

    def test_mahalanobis_rbf_kernel(self):
        config = self.config.copy()
        config["M0"] = torch.diag(1 * torch.ones(self.n))
        kernel = MahalanobisRBFKernel(**config)
        K = kernel._nonkeops_forward(self.x, self.x).detach().numpy()
        assert np.allclose(K[0, :], [1, 0.1353353, 0.1353353, 0.01831564], atol=1e-3)
        assert np.allclose(np.diag(K), 1.0, atol=1e-3)
        assert np.allclose(K, K.T, atol=1e-4)

        K2 = kernel._keops_forward(self.x, self.x).to_dense().detach().numpy()
        assert np.allclose(K, K2, atol=1e-4)

        config["M0"] = torch.diag(0.5 * torch.ones(self.n))
        kernel = MahalanobisRBFKernel(**config)
        K = kernel._nonkeops_forward(self.x, self.x).detach().numpy()
        assert np.allclose(K[0, :], [1, 0.36787945, 0.36787945, 0.1353353], atol=1e-3)
        assert np.allclose(np.diag(K), 1.0, atol=1e-3)
        assert np.allclose(K, K.T, atol=1e-4)

        K2 = kernel._keops_forward(self.x, self.x).to_dense().detach().numpy()
        assert np.allclose(K, K2, atol=1e-4)

        # Check longer sequences
        alphabet = list("ACBD")
        sl = 6
        kernel = MahalanobisRBFKernel(alphabet=alphabet, seq_length=sl)
        x = get_full_space_one_hot(sl, len(alphabet))
        K = kernel._nonkeops_forward(x, x).detach()
        cov1 = K.numpy()
        cov2 = kernel._keops_forward(x, x).detach().to_dense().numpy()
        assert np.allclose(cov2, cov1)

        # Ensure PSD
        for _ in range(10):
            v = np.random.normal(size=K.shape[1])
            assert np.dot(v, K @ v) >= 0.0

        K = kernel.forward(x, x, diag=True).detach().numpy()
        assert np.allclose(K, 1.0)

        # Ensure diagonal is properly computed
        x1, x2 = x[:10, :], x[10:20, :]
        K = kernel.forward(x1, x2).detach().numpy()
        diag = kernel.forward(x1, x2, diag=True).detach().numpy()
        assert np.allclose(diag, np.diag(K))

    def test_factor_analysis_kernel(self):
        # Initialize to exponential kernel
        config = self.config.copy()
        config["ndim"] = 4
        config["A0"] = torch.eye(4)
        kernel = FactorAnalysisKernel(**config)

        K = kernel._nonkeops_forward(self.x, self.x).detach().numpy()
        print(K[0, :])
        assert np.allclose(K[0, :], [1, 0.1353353, 0.1353353, 0.01831564], atol=1e-3)
        assert np.allclose(np.diag(K), 1.0, atol=1e-3)
        assert np.allclose(K, K.T, atol=1e-4)

        K2 = kernel._keops_forward(self.x, self.x).to_dense().detach().numpy()
        assert np.allclose(K, K2, atol=1e-4)

        # Initialize with other values
        config["ndim"] = 1
        config["A0"] = torch.Tensor([[-0.5], [0.5], [0.0], [0]])
        kernel = FactorAnalysisKernel(**config)
        K = kernel._nonkeops_forward(self.x, self.x).detach().numpy()
        assert np.allclose(K[0, :], [1, np.exp(-1), 1, np.exp(-1)], atol=1e-3)
        assert np.allclose(np.diag(K), 1.0, atol=1e-3)
        assert np.allclose(K, K.T, atol=1e-4)

        K2 = kernel._keops_forward(self.x, self.x).to_dense().detach().numpy()
        assert np.allclose(K, K2, atol=1e-4)

    def test_connectedness_FA_kernel(self):
        config = self.config.copy()
        config["C0"] = torch.Tensor([[1, 0.0], [0.0, 1]])

        # Test equivalence to Connectedness model
        kernel = ConnectednessFactorAnalysisKernel(**config)
        K = kernel._nonkeops_forward(self.x, self.x).detach().numpy()
        print(K)
        assert np.allclose(K[0, :], [1, 0.1353353, 0.1353353, 0.01831564], atol=1e-3)
        assert np.allclose(np.diag(K), 1.0, atol=1e-3)
        assert np.allclose(K, K.T, atol=1e-4)

        K2 = kernel._keops_forward(self.x, self.x).to_dense().detach().numpy()
        assert np.allclose(K, K2, atol=1e-4)

        C = kernel.get_C().detach().numpy()
        assert np.allclose(C, config["C0"])

        # Test with non-zero off diagonal terms
        config["C0"] = torch.Tensor([[1, 0.2], [0.2, 1]])
        kernel = ConnectednessFactorAnalysisKernel(**config)
        K = kernel._nonkeops_forward(self.x, self.x).detach().numpy()
        print(K)

        x = np.array(self.x)
        x1 = x[[1]]
        x2 = x[[2]]
        print(x)
        M = np.array([[1, 0, 0.2, 0], [0, 1, 0, 0.2], [0.2, 0, 1, 0], [0, 0.2, 0, 1]])
        k = np.exp(x @ M @ x.T)
        print(x1)
        print(x1 @ M @ x1.T)
        print(x2)
        print(x2 @ M @ x2.T)
        print(x1.T @ x1)
        print(x2.T @ x2)
        print(x.T @ M @ x)

        print(k)
        print([k[0, 0] / k[0, 1], k[0, 2] / k[0, 3]])
        exit()
        exp_K = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                x1 = np.array(self.x[i])
                x2 = np.array(self.x[j])
                d = x2 - x1
                exp_K[i, j] = np.exp(-d.T @ M @ d)
                print(i, j, d, d.T @ M @ d)
                print((M @ d) * d)
        print(exp_K)
        assert np.allclose(K, exp_K)

        K2 = kernel._keops_forward(self.x, self.x).to_dense().detach().numpy()
        assert np.allclose(K, K2, atol=1e-4)

    def test_diploid_kernel(self):
        config = self.config.copy()
        config.update(
            {
                "log_lambda0": torch.Tensor([np.log(0.5)]),
                "log_eta0": torch.Tensor([np.log(0.5)]),
                "logit_p0": torch.Tensor([0.0]),
            }
        )
        kernel = DiploidKernel(**config)
        x = get_full_space_one_hot(2, 3)

        # Check kernel calculation
        cov = kernel.forward(x, x).detach().numpy()
        assert np.allclose(cov[0, 0], 1)
        assert np.allclose(cov[0, 1], 1 / 5.0)
        assert np.allclose(cov[0, 2], 1 / 5.0)
        assert np.allclose(cov[1, 1], 3 / 5.0)

        diag = kernel.forward(x, x, diag=True).detach().numpy()
        assert np.allclose(diag, np.diag(cov))

        # Check different parameters
        config.update(
            {
                "log_lambda0": torch.Tensor([np.log(0.5)]),
                "log_eta0": torch.Tensor([np.log(0.25)]),
                "logit_p0": torch.Tensor([-np.log(3)]),
            }
        )
        kernel = DiploidKernel(**config)
        cov = kernel.forward(x, x).detach().numpy()

        assert np.allclose(np.diag(cov), 1.0)
        assert np.allclose(cov[0, 1], 3 / 7.0)
        assert np.allclose(cov[0, 2], 5 / 21.0)
        
        # Check with partitioning
        config.update({"partition_size": 1})
        kernel = DiploidKernel(**config)
        cov = (kernel.forward(x, x) @ torch.eye(x.shape[0])).detach().numpy()

        assert np.allclose(np.diag(cov), 1.0)
        assert np.allclose(cov[0, 1], 3 / 7.0)
        assert np.allclose(cov[0, 2], 5 / 21.0)


if __name__ == "__main__":
    import sys

    sys.argv = ["", "KernelsTests"]
    unittest.main()
