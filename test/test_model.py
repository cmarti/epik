#!/usr/bin/env python
import sys
import unittest
from os.path import join
from subprocess import check_call
from tempfile import NamedTemporaryFile

import gpytorch
import numpy as np
import pandas as pd
import torch
from itertools import product
from scipy.stats import pearsonr, multivariate_normal
from scipy.special import comb
from gpytorch.distributions import MultivariateNormal
from torch.distributions.transforms import CorrCholeskyTransform
from torch.distributions import Dirichlet

from epik.kernel import (
    AdditiveKernel,
    ConnectednessKernel,
    GeometricKernel,
    GeneralProductKernel,
    JengaKernel,
    PairwiseKernel,
    VarianceComponentKernel,
    FactorAnalysisKernel,
)
from epik.model import EpiK
from epik.settings import BIN_DIR, KERNELS
from epik.utils import (
    get_full_space_one_hot,
    get_mut_effs_contrast_matrix,
    one_hot_to_seq,
    seq_to_one_hot,
    get_one_hot_encoding,
    get_random_sequences,
)


class ModelsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.log_lambdas0 = torch.tensor([-5, 2.0, 1, -1.5, -3])
        self.alphabet = np.array(["A", "C", "G", "T"]).tolist()
        self.alleles = "".join(self.alphabet)
        self.alpha = len(self.alphabet)
        self.l = self.log_lambdas0.shape[0] - 1
        self.alphabet_list = [self.alphabet] * self.l
        self.add_size = (self.alpha - 1) * self.l
        self.pw_size = int((self.alpha - 1) ** 2 * comb(self.l, 2))
        self.X = np.array(["".join(x) for x in product(*self.alphabet_list)])
        self.sigma = 0.2
        self.ptrain = 0.8

        self.kernel = VarianceComponentKernel(
            alphabet_list=self.alphabet_list, log_lambdas0=self.log_lambdas0
        )
        self.model = EpiK(self.kernel, alphabet_list=self.alphabet_list)
        dataset = self.model.simulate_dataset(
            self.X, sigma=self.sigma, ptrain=self.ptrain
        )
        self.X_train, self.y_train, self.X_test, self.y_test, self.y_var = dataset
        self.y_test = self.y_test.numpy()

        k = np.array(
            [
                [1, 0.6, 0.4, 0.1],
                [0.6, 1, 0.2, 0.7],
                [0.4, 0.3, 1, 0.8],
                [0.2, 0.7, 0.8, 1],
            ]
        )
        ps = np.linspace(0.1, 2.5, self.l)
        ks = [(p * np.eye(4) + k) / (1 + p) for p in ps]
        t = CorrCholeskyTransform()
        self.theta0 = [t._inverse(torch.Tensor(k)) for k in ks]
        self.vc_kernels = [
            AdditiveKernel,
            PairwiseKernel,
            VarianceComponentKernel,
        ]
        self.product_kernels = [
            GeometricKernel,
            ConnectednessKernel,
            JengaKernel,
            GeneralProductKernel,
        ]

    def test_gaussian(self):
        C = np.array(
            [
                [1.0, 0.5, 0.5, 0.25],
                [0.5, 1.0, 0.25, 0.5],
                [0.5, 0.25, 1.0, 0.5],
                [0.25, 0.5, 0.5, 1.0],
            ]
        )

        # Test scipy implementation
        gaussian1 = multivariate_normal(mean=np.zeros(4), cov=C)
        s1 = gaussian1.rvs(size=5000)
        c1 = np.corrcoef(s1.T)
        assert np.allclose(c1, C, atol=0.05)

        # Test gpytorch Cholesky implementation
        with gpytorch.settings.fast_computations(covar_root_decomposition=False):
            gaussian1 = MultivariateNormal(torch.zeros(4), torch.Tensor(C))
            s2 = gaussian1.sample(sample_shape=torch.Size([5000])).numpy()
            c2 = np.corrcoef(s2.T)
            assert np.allclose(c2, C, atol=0.05)

        # Test gpytorch Tridiagonal implementation
        # Some problem with a too high tolerance in the lanczos decomposition
        with gpytorch.settings.fast_computations(
            covar_root_decomposition=True
        ), gpytorch.settings.max_cholesky_size(
            1
        ), gpytorch.settings.max_root_decomposition_size(
            10
        ), gpytorch.settings.cg_tolerance(1e-16):
            gaussian1 = MultivariateNormal(torch.zeros(4), torch.Tensor(C))
            s3 = gaussian1.sample(sample_shape=torch.Size([5000])).numpy()
            c3 = np.corrcoef(s3.T)
            assert np.allclose(c3, C, atol=0.05)

    def test_calc_mll(self):
        np.random.seed(0)
        seq_length = 8
        theta0 = torch.Tensor(np.geomspace(0.1, 0.8, seq_length))
        with torch.no_grad():
            kernel = ConnectednessKernel(
                seq_length=seq_length, alphabet=self.alphabet, theta0=theta0
            )
            for n in [500, 1000, 2000]:
                seqs = get_random_sequences(
                    n=n, seq_length=seq_length, alphabet=self.alphabet
                )
                X = get_one_hot_encoding(seqs, alphabet_list=kernel.alphabet_list)
                y_var = 0.4 * np.ones(n)
                Sigma = kernel(X, X).to_dense().numpy() + np.diag(y_var)
                gaussian = multivariate_normal(np.zeros(n), Sigma)

                # Sample from gaussian
                y = gaussian.rvs()

                # Compute log-probability with scipy
                logp1 = gaussian.logpdf(y)

                model = EpiK(kernel, seq_length=seq_length, alphabet=self.alphabet)
                model.set_data(X=seqs, y=y, y_var=y_var)

                # Compute log-probability with chokesly decomposition
                logp2 = model.calc_mll(method="cholesky").item()
                assert np.allclose(logp1, logp2, atol=1e-2)

                # Compute log-probability with cg approach
                for _ in range(5):
                    logp2 = model.calc_mll(
                        method="cg", cg_tol=0.1, n_lanczos_iter=200, n_trace_samples=50
                    ).item()
                    assert np.allclose(logp1, logp2, rtol=0.1)

    def test_diagnose_mll(self):
        model = EpiK(alphabet=self.alphabet, seq_length=self.l, kernel="Connectedness")
        model.set_data(X=self.X_train, y=self.y_train, y_var=self.y_var)
        mll_diagnosis = model.diagnose_mll()

        assert mll_diagnosis.shape[0] == 150
        columns = [
            "method",
            "cg_tol",
            "max_cg_iter",
            "n_lanczos_iter",
            "n_trace_samples",
            "preconditioner_size",
            "mll",
            "param",
        ]
        assert np.all(mll_diagnosis.columns == columns)

    def test_simulate(self):
        model = EpiK(alphabet=self.alphabet, seq_length=self.l, kernel=self.kernel)
        x = model.encode(self.X).numpy()
        distance = self.l - x @ x.T

        f1 = model.simulate(self.X, n=10000, method="cholesky").numpy()
        cors1 = pd.DataFrame(f1).corr().values

        f2 = model.simulate(
            self.X, n=10000, method="lanczos", root_decomposition_size=200
        ).numpy()
        cors2 = pd.DataFrame(f2).corr().values

        for d in range(self.l + 1):
            d_idx = distance == d
            n = d_idx.sum()
            c1 = np.sum(d_idx * cors1) / n
            c2 = np.sum(d_idx * cors2) / n
            assert np.allclose(c1, c2, rtol=0.1)

    def test_predict(self):
        model = EpiK(alphabet=self.alphabet, seq_length=self.l, kernel=self.kernel)
        model.set_data(self.X_train, self.y_train, self.y_var)

        # Predict on test data without variance
        results1 = model.predict(self.X_test, calc_variance=False)
        r2 = pearsonr(results1["coef"], self.y_test)[0] ** 2
        assert r2 > 0.75

        # Predict on test data with variance
        results2 = model.predict(self.X_test, calc_variance=True)
        assert np.allclose(results2["coef"], results1["coef"])

        # Check interval coverage
        bound1 = results2["lower_ci"] < self.y_test
        bound2 = results2["upper_ci"] > self.y_test
        coverage = np.mean(bound1 & bound2)
        assert coverage > 0.85

    def test_contrasts(self):
        model = EpiK(alphabet=self.alphabet, seq_length=self.l, kernel=self.kernel)
        model.set_data(self.X_train, self.y_train, self.y_var)

        # Define target sequences and contrast
        X = ["ACGT", "ACGG", "TCGT", "TCGG"]
        labels = ["A0T_T3G"]
        contrast_matrix = pd.DataFrame([[1, -1, -1, 1]], index=labels, columns=X)

        # Make contrast
        results1 = model.make_contrasts(contrast_matrix, calc_variance=True)
        assert results1.shape == (1, 4)

        results = model.predict_mut_effects(seq0="ACGT", calc_variance=True)
        assert results.shape == (self.add_size, 4)

        results = model.predict_epistatic_coeffs(seq0="ACGT", calc_variance=True)
        assert results.shape == (self.pw_size, 4)
        assert np.allclose(results1, results.loc[labels, :], atol=1e-3)
    
    def test_kronecker_map_projection(self):
        model = EpiK(
            alphabet=self.alphabet,
            seq_length=self.l,
            kernel="GeneralProduct",
            kernel_kwargs={"theta0": self.theta0},
        )
        data = model.simulate_dataset(self.X, sigma=0.1, ptrain=0.9, method="cholesky")
        X_train, y_train, _, _, y_train_var = data
        model.set_data(X_train, y_train, y_train_var)

        # Compute f mean directly
        f = model.get_posterior(self.X)
        f_mean1 = f.mean.mean().item()

        # Compute f mean with kronecker factorization
        m = torch.full((self.alpha, self.alpha), 1.0 / self.alpha)
        matrices = [m] * self.l
        x_p = torch.zeros(self.alpha)
        x_p[0] = 1
        x1 = torch.hstack([x_p] * self.l).unsqueeze(0)
        f_mean2 = model.calc_kron_dot_map(x1, matrices).item()
        assert np.allclose(f_mean1, f_mean2)

        # Test with arbitrary matrices
        sizes = np.random.randint(low=2, high=5, size=self.l)
        alphabet = list('ABCDE')
        alphabet_list = [alphabet[:s] for s in sizes]
        seqs = np.array(["".join(x) for x in product(*alphabet_list)])
        x1 = get_one_hot_encoding(seqs, alphabet_list)
        matrices = []
        matrix = np.array([[1.]])
        for s in sizes:
            m = np.random.normal(size=(s, self.alpha))
            matrix = np.kron(matrix, m)
            matrices.append(torch.Tensor(m))
        matrix = torch.Tensor(matrix)

        v1 = matrix @ f.mean
        v2 = model.calc_kron_dot_map(x1, matrices)
        assert np.allclose(v1, v2, atol=1e-4)
    
    def test_calc_gauge_fixed_add_theta(self):
        model = EpiK(
            alphabet=self.alphabet,
            seq_length=self.l,
            kernel="GeneralProduct",
            kernel_kwargs={"theta0": self.theta0},
        )
        data = model.simulate_dataset(self.X, sigma=0.1, ptrain=0.9, method="cholesky")
        X_train, y_train, _, _, y_train_var = data
        model.set_data(X_train, y_train, y_train_var)

        dist = Dirichlet(torch.ones(self.alpha))
        pi_lc = [dist.sample() for _ in range(self.l)]
        P0s = [pi_p.unsqueeze(0) for pi_p in pi_lc]
        P1s = [torch.eye(P0_p.shape[1]) - P0_p for P0_p in P0s]
        P = []
        for pos in range(self.l):
            matrices = [
                P1_p if p == pos else P0_p
                for p, (P0_p, P1_p) in enumerate(zip(P0s, P1s))
            ]
            matrix = np.array([[1.0]])
            for m in matrices:
                matrix = np.kron(matrix, m)
            P.append(torch.Tensor(matrix))
        P = torch.vstack(P)

        # Compute theta directly from MAP
        f = model.get_posterior(self.X).mean
        theta1 = P @ f

        # Compute f mean with kronecker factorization
        theta2 = model.calc_gauge_fixed_add_theta(pi_lc)
        assert np.allclose(theta1, theta2["theta"], atol=1e-4)
    
    def test_kronecker_map_quad(self):
        model = EpiK(
            alphabet=self.alphabet,
            seq_length=self.l,
            kernel="GeneralProduct",
            kernel_kwargs={"theta0": self.theta0},
        )
        data = model.simulate_dataset(self.X, sigma=0.1, ptrain=0.9, method="cholesky")
        X_train, y_train, _, _, y_train_var = data
        model.set_data(X_train, y_train, y_train_var)

        # Compute f norm directly
        f = model.get_posterior(self.X)
        f_norm1 = (f.mean ** 2).sum().item()

        # Compute f norm with kronecker factorization
        m = torch.eye(self.alpha)
        matrices = [m] * self.l
        f_norm2 = model.calc_kron_quad_map(matrices)
        assert np.allclose(f_norm1, f_norm2, atol=1e-2)

        # Test with arbitrary matrices
        matrix = np.array([[1.0]])
        Q = np.array([[1.0]])
        matrices = []
        qs = []

        for _ in range(self.l):
            q = np.random.normal(size=(self.alpha, self.alpha))
            A_p = q @ q.T
            matrix = np.kron(matrix, A_p)
            matrices.append(torch.Tensor(A_p))
            Q = np.kron(Q, q.T)
            qs.append(torch.Tensor(q.T))
        matrix = torch.Tensor(matrix)
        Q = torch.Tensor(Q)

        v1 = torch.dot(f.mean, matrix @ f.mean).item()
        v2 = model.calc_kron_quad_map(matrices)
        assert np.allclose(v1, v2, atol=1e-4)

        # Test consistency with kron_dot
        u = Q @ f.mean
        v3 = torch.dot(u, u).item()
        assert np.allclose(v1, v3, atol=1e-4)

        x1 = get_one_hot_encoding(self.X, self.alphabet_list)
        u = model.calc_kron_dot_map(x1, qs)
        v4 = torch.dot(u, u).item()
        assert np.allclose(v1, v4, atol=1e-4)

    def test_fit(self):
        model = EpiK(alphabet=self.alphabet, seq_length=self.l, kernel="VC")
        model.set_data(self.X_train, self.y_train, self.y_var)
        model.fit(n_iter=500, learning_rate=0.005, cg_tol=0.1, n_lanczos_iter=100)
        log_lambdas = model.kernel.log_lambdas.detach().cpu().numpy().flatten()
        r = pearsonr(log_lambdas[1:], self.log_lambdas0[1:])[0]
        assert r > 0.8

    def test_fit_predict_vc_kernels(self):
        prev_mll = -np.inf
        r2_bounds = [0.0, 0.3, 0.5]
        kwargs = {
            "n_iter": 100,
            "learning_rate": 0.05,
            "track_progress": True,
            "cg_tol": 0.1,
            "n_lanczos_iter": 200,
        }
        for kernel, r2_bound in zip(self.vc_kernels, r2_bounds):
            # Infer hyperparameters with Cholesky decomposition
            k = kernel(alphabet=self.alphabet, seq_length=self.l, use_keops=False)
            model = EpiK(kernel=k, alphabet=self.alphabet, seq_length=self.l)
            model.set_data(self.X_train, self.y_train, self.y_var)
            model.fit(mll_method="cholesky", **kwargs)
            assert model.mll >= prev_mll
            prev_mll = model.mll

            test_y_pred = model.predict(self.X_test, method='cholesky')["coef"]
            r2 = pearsonr(test_y_pred, self.y_test)[0] ** 2
            assert r2 > r2_bound

            # Infer hyperparameters with CG
            k = kernel(alphabet=self.alphabet, seq_length=self.l, use_keops=False)
            model = EpiK(kernel=k, alphabet=self.alphabet, seq_length=self.l)
            model.set_data(self.X_train, self.y_train, self.y_var)
            model.fit(mll_method="cg", **kwargs)
            assert np.allclose(model.mll, prev_mll, rtol=0.05)

            test_y_pred = model.predict(self.X_test, method='cg')["coef"]
            r2 = pearsonr(test_y_pred, self.y_test)[0] ** 2
            assert r2 > r2_bound

            # Infer hyperparameters with CG and KeOps
            k = kernel(alphabet=self.alphabet, seq_length=self.l, use_keops=True)
            model = EpiK(kernel=k, alphabet=self.alphabet, seq_length=self.l)
            model.set_data(self.X_train, self.y_train, self.y_var)
            model.fit(mll_method="cg", **kwargs)
            assert np.allclose(model.mll, prev_mll, rtol=0.05)

            test_y_pred = model.predict(self.X_test, method="cg")["coef"]
            r2 = pearsonr(test_y_pred, self.y_test)[0] ** 2
            assert r2 > r2_bound

    def test_fit_predict_product_kernels(self):
        model = EpiK(
            alphabet=self.alphabet,
            seq_length=self.l,
            kernel="GeneralProduct",
            kernel_kwargs={"theta0": self.theta0},
        )
        data = model.simulate_dataset(self.X, sigma=0.1, ptrain=0.9, method="cholesky")
        X_train, y_train, test_x, test_y, y_train_var = data

        prev_mll = -np.inf
        kwargs = {
            "n_iter": 500,
            "learning_rate": 0.01,
            "track_progress": True,
            "cg_tol": 0.1,
            "n_lanczos_iter": 100,
        }
        r2_bound = 0.4
        for kernel in self.product_kernels:
            # Infer hyperparameters with Cholesky decomposition
            k = kernel(alphabet=self.alphabet, seq_length=self.l, use_keops=False)
            model = EpiK(kernel=k, alphabet=self.alphabet, seq_length=self.l)
            model.set_data(X_train, y_train, y_train_var)
            model.fit(mll_method="cholesky", **kwargs)
            assert model.mll >= prev_mll
            prev_mll = model.mll

            test_y_pred = model.predict(test_x, method='cholesky')["coef"]
            r2 = pearsonr(test_y_pred, test_y)[0] ** 2
            assert r2 > r2_bound

            # Infer hyperparameters with CG
            k = kernel(alphabet=self.alphabet, seq_length=self.l, use_keops=False)
            model = EpiK(kernel=k, alphabet=self.alphabet, seq_length=self.l)
            model.set_data(X_train, y_train, y_train_var)
            model.fit(mll_method="cg", **kwargs)
            assert np.allclose(model.mll, prev_mll, atol=10)

            test_y_pred = model.predict(test_x, method="cg")["coef"]
            r2 = pearsonr(test_y_pred, test_y)[0] ** 2
            assert r2 > r2_bound

            # Infer hyperparameters with CG and KeOps
            k = kernel(alphabet=self.alphabet, seq_length=self.l, use_keops=True)
            model = EpiK(kernel=k, alphabet=self.alphabet, seq_length=self.l)
            model.set_data(X_train, y_train, y_train_var)
            model.fit(mll_method="cg", **kwargs)
            assert np.allclose(model.mll, prev_mll, atol=10)

            test_y_pred = model.predict(test_x, method="cg")["coef"]
            r2 = pearsonr(test_y_pred, test_y)[0] ** 2
            assert r2 > r2_bound

    def xtest_bin(self):
        bin_fpath = join(BIN_DIR, "EpiK.py")

        # Simulate data
        train_seqs = one_hot_to_seq(self.X_train.numpy(), self.alphabet)
        test_seqs = one_hot_to_seq(self.X_test.numpy(), self.alphabet)
        data = pd.DataFrame({"y": self.y_train.numpy()}, index=train_seqs)
        test = pd.DataFrame({"x": test_seqs})

        with NamedTemporaryFile() as fhand:
            out_fpath = fhand.name
            params_fpath = "{}.model_params.pth".format(out_fpath)
            data_fpath = "{}.train.csv".format(out_fpath)
            xpred_fpath = "{}.test.csv".format(out_fpath)
            data.to_csv(data_fpath)
            test.to_csv(xpred_fpath, header=False, index=False)

            for label in KERNELS:
                # Fit hyperparameters
                cmd = [
                    sys.executable,
                    bin_fpath,
                    data_fpath,
                    "-k",
                    label,
                    "-o",
                    out_fpath,
                    "-n",
                    "50",
                ]
                check_call(cmd)

                # Predict test sequences
                cmd = [
                    sys.executable,
                    bin_fpath,
                    data_fpath,
                    "-k",
                    label,
                    "-o",
                    out_fpath,
                    "-n",
                    "0",
                    "-p",
                    xpred_fpath,
                    "--params",
                    params_fpath,
                    "--calc_variance",
                ]
                check_call(cmd)

                # Calculate mutational effects contrasts
                cmd = [
                    sys.executable,
                    bin_fpath,
                    data_fpath,
                    "-k",
                    label,
                    "-o",
                    out_fpath,
                    "-n",
                    "0",
                    "-s",
                    self.seq0,
                    "--params",
                    params_fpath,
                    "--calc_variance",
                ]
                check_call(cmd)

    def xtest_FA(self):
        # Simulate data
        k1 = FactorAnalysisKernel(
            n_alleles=self.alpha, seq_length=self.l, ndim=2, train_sigma2=True
        )
        k2 = JengaKernel(n_alleles=self.alpha, seq_length=self.l)
        model = EpiK(k1, track_progress=True)
        y = model.simulate(self.X_train).flatten()

        for _ in range(3):
            k1 = FactorAnalysisKernel(
                n_alleles=self.alpha, seq_length=self.l, ndim=2, train_sigma2=True
            )
            k2 = JengaKernel(n_alleles=self.alpha, seq_length=self.l)
            with gpytorch.settings.max_cholesky_size(5000):
                model = EpiK(k1, track_progress=True)
                model.set_data(self.X_train, y, self.y_var)
                model.fit(n_iter=1000, learning_rate=0.1)

            M1 = k1.get_M().detach().numpy()
            l1, q1 = np.linalg.eigh(M1)
            print(l1)
            # print(k1.get_diag())
            # print(q1[:, -2:].T @ q1[:, -2:])
            print(k1.get_delta().detach().numpy())


if __name__ == "__main__":
    import sys

    sys.argv = ["", "ModelsTests"]
    unittest.main()
