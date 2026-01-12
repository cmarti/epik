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
from scipy.stats import pearsonr, multivariate_normal
from gpytorch.distributions import MultivariateNormal

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
    encode_seqs,
    get_random_sequences,
)


class ModelsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.log_lambdas0 = torch.tensor([-5, 2.0, 1, -1.5, -3])
        self.alphabet = np.array(["A", "C", "G", "T"])
        self.alleles = "".join(self.alphabet)
        self.alpha = len(self.alphabet)
        self.l = self.log_lambdas0.shape[0] - 1
        self.add_size = (self.alpha - 1) * self.l
        self.seq0 = "A" * self.l
        self.X = get_full_space_one_hot(seq_length=self.l, n_alleles=self.alpha)
        self.sigma = 0.2
        self.ptrain = 0.8

        self.kernel = VarianceComponentKernel(
            n_alleles=self.alpha, seq_length=self.l, log_lambdas0=self.log_lambdas0
        )
        self.model = EpiK(self.kernel)
        dataset = self.model.simulate_dataset(
            self.X, sigma=self.sigma, ptrain=self.ptrain
        )
        self.X_train, self.y_train, self.X_test, self.y_test, self.y_var = dataset
        self.y_test = self.y_test.numpy()

        self.vc_kernels = [
            AdditiveKernel,
            PairwiseKernel,
            GeometricKernel,
            VarianceComponentKernel,
        ]
        self.product_kernels = [
            GeometricKernel,
            ConnectednessKernel,
            JengaKernel,
            GeneralProductKernel,
        ]
    
    def test_gaussian(self):
        C = np.array([[1.0, 0.5, 0.5, 0.25],
                      [0.5, 1.0, 0.25, 0.5],
                      [0.5, 0.25, 1.0, 0.5],
                      [0.25, 0.5, 0.5, 1.0]])
        
        # Test scipy implementation
        gaussian1 = multivariate_normal(mean=np.zeros(4), cov=C)
        s1 = gaussian1.rvs(size=5000)
        c1 = np.corrcoef(s1.T)
        assert( np.allclose(c1, C, atol=0.05))
        
        # Test gpytorch Cholesky implementation
        with gpytorch.settings.max_cholesky_size(100):
            gaussian1 = MultivariateNormal(torch.zeros(4), torch.Tensor(C))
            s2 = gaussian1.sample(sample_shape=torch.Size([5000])).numpy()
            c2 = np.corrcoef(s2.T)
            assert( np.allclose(c2, C, atol=0.05))
        
        # Test gpytorch Tridiagonal implementation
        with gpytorch.settings.max_cholesky_size(1), gpytorch.settings.max_root_decomposition_size(10), gpytorch.settings.cg_tolerance(1e-16):
            gaussian1 = MultivariateNormal(torch.zeros(4), torch.Tensor(C))
            s3 = gaussian1.sample(sample_shape=torch.Size([5000])).numpy()
            c3 = np.corrcoef(s3.T)
            assert( np.allclose(c3, C, atol=0.05))
        

    def test_calc_mll(self):
        seq_length = 8
        with torch.no_grad():
            kernel = ConnectednessKernel(n_alleles=self.alpha, seq_length=seq_length)
            for n in [500, 1000, 2000]:
                seqs = get_random_sequences(
                    n=n, seq_length=seq_length, alphabet=self.alphabet
                )
                X = encode_seqs(seqs, alphabet=self.alphabet)
                y_var = 0.1 * np.ones(n)
                D = np.diag(y_var)
                mu = np.zeros(n)
                Sigma = kernel(X, X).to_dense().numpy() + D
                gaussian = multivariate_normal(mu, Sigma)
                y = gaussian.rvs()

                # Compute log-probability with scipy
                logp1 = gaussian.logpdf(y)

                model = EpiK(kernel)
                model.set_data(X=X, y=y, y_var=y_var)

                # Compute log-probability with chokesly decomposition
                logp2 = model.calc_mll(method='cholesky').item()
                assert np.allclose(logp1, logp2, atol=1e-4)

                # Compute log-probability with cg approach
                for _ in range(5):
                    logp2 = model.calc_mll(
                        method="cg", n_lanczos_iter=100, n_trace_samples=100
                    ).item()
                    assert np.allclose(logp1, logp2, atol=20)

    def test_diagnose_mll(self):
        kernel = ConnectednessKernel(n_alleles=self.alpha, seq_length=self.l)
        model = EpiK(kernel, method='cg')
        model.set_data(X=self.X_train, y=self.y_train, y_var=self.y_var)
        mll_diagnosis = model.diagnose_mll()

        assert mll_diagnosis.shape[0] == 150
        columns = ["param", "n_lanczos", 'cg_tol', 'n_trace_samples', "mll"]
        assert np.all(mll_diagnosis.columns == columns)

    def test_simulate_cholesky(self):
        sl, a, lambdas0 = 2, 2, torch.log(torch.tensor([2, 1, 0.2]))
        X = get_full_space_one_hot(seq_length=sl, n_alleles=a)
        kernel = VarianceComponentKernel(
            n_alleles=a, seq_length=sl, log_lambdas0=lambdas0
        )
        k = kernel(X, X).to_dense().detach().numpy()
        
        model = EpiK(kernel, method='cholesky')
        y = pd.DataFrame(model.simulate(X, n=10000).numpy())
        cors = y.corr().values
        rho1 = np.array(
            [
                cors[0, 1],
                cors[0, 2],
                cors[1, 0],
                cors[1, 3],
                cors[2, 0],
                cors[2, 3],
                cors[3, 1],
                cors[3, 2],
            ]
        )
        rho2 = np.array([cors[0, 3], cors[1, 2], cors[2, 1], cors[3, 0]])
        assert rho1.std() < 0.2
        assert rho2.std() < 0.1
        assert np.allclose(cors, k, atol=0.05)
    
    def test_simulate_cg(self):
        sl, a, lambdas0 = 2, 2, torch.log(torch.tensor([2, 1, 0.2]))
        X = get_full_space_one_hot(seq_length=sl, n_alleles=a)
        kernel = VarianceComponentKernel(
            n_alleles=a, seq_length=sl, log_lambdas0=lambdas0
        )
        k = kernel(X, X).to_dense().detach().numpy()
        
        model = EpiK(kernel, method='cg', n_lanczos_iter=100)
        y = pd.DataFrame(model.simulate(X, n=10000).numpy())
        cors = y.corr().values

        rho1 = np.array(
            [
                cors[0, 1],
                cors[0, 2],
                cors[1, 0],
                cors[1, 3],
                cors[2, 0],
                cors[2, 3],
                cors[3, 1],
                cors[3, 2],
            ]
        )
        rho2 = np.array([cors[0, 3], cors[1, 2], cors[2, 1], cors[3, 0]])
        # assert rho1.std() < 0.2
        # assert rho2.std() < 0.1
        msg = f'CG simulation {cors} does not match kernel matrix {k}'
        assert np.allclose(cors, k, atol=0.05), msg

    def test_predict(self):
        # Predict on test data
        model = EpiK(self.kernel)
        model.set_data(self.X_train, self.y_train, self.y_var)
        results1 = model.predict(self.X_test, calc_variance=False)
        r2 = pearsonr(results1["coef"], self.y_test)[0] ** 2
        assert r2 > 0.6

        # Predict on test data with variance
        model = EpiK(self.kernel)
        model.set_data(self.X_train, self.y_train, self.y_var)
        results2 = model.predict(self.X_test, calc_variance=True)
        assert np.allclose(results2["coef"], results1["coef"])

        # Check calibration
        bound1 = results2["lower_ci"] < self.y_test
        bound2 = results2["upper_ci"] > self.y_test
        calibration = np.mean(bound1 & bound2)
        assert calibration > 0.9

    def test_contrast(self):
        # Define target sequences and contrast
        test_x = torch.tensor(
            [
                [1, 0, 0, 0, 1, 0, 0, 0] + [1, 0, 0, 0] * 2,
                [0, 1, 0, 0, 1, 0, 0, 0] + [1, 0, 0, 0] * 2,
                [1, 0, 0, 0, 0, 1, 0, 0] + [1, 0, 0, 0] * 2,
                [0, 1, 0, 0, 0, 1, 0, 0] + [1, 0, 0, 0] * 2,
            ]
        )
        contrast_matrix = torch.tensor([[1, -1, -1, 1]])

        # Define model
        model = EpiK(self.kernel, track_progress=False)
        model.set_data(self.X_train, self.y_train, self.y_var)

        # Make contrast
        m, cov = model.make_contrasts(contrast_matrix, test_x, calc_variance=True)
        assert m.shape == (1,)
        assert cov.shape == (1, 1)

        # Make contrast with built-in matrix functions
        contrast_matrix = get_mut_effs_contrast_matrix(
            seq0=self.seq0, alleles=self.alleles
        )
        test_x = seq_to_one_hot(contrast_matrix.columns, alleles=self.alleles)
        contrast_matrix = torch.Tensor(contrast_matrix.values)
        m, cov = model.make_contrasts(contrast_matrix, test_x, calc_variance=True)
        assert m.shape == (self.add_size,)
        assert cov.shape == (self.add_size, self.add_size)

        # Make contrast with built-in method
        results = model.predict_mut_effects(
            seq0=self.seq0, alleles=self.alleles, calc_variance=False
        )
        assert results.shape == (self.add_size, 1)
        assert np.allclose(results["coef"], m.numpy())

        # Make contrast with built-in method and variance
        results = model.predict_mut_effects(
            seq0=self.seq0, alleles=self.alleles, calc_variance=True
        )
        assert results.shape == (self.add_size, 4)
        assert np.allclose(results["coef"], m.numpy())
        assert np.allclose(results["stderr"], np.sqrt(np.diag(cov.numpy())))

    def test_fit(self):
        kernel = VarianceComponentKernel(n_alleles=self.alpha, seq_length=self.l)
        model = EpiK(kernel, track_progress=True, method='cg')
        model.set_data(self.X_train, self.y_train, self.y_var)
        model.fit(n_iter=100, learning_rate=0.01)
        log_lambdas = kernel.log_lambdas.detach().cpu().numpy().flatten()
        r = pearsonr(log_lambdas[1:], self.log_lambdas0[1:])[0]
        assert r > 0.6
        
        assert len(model.params_history) == 100
        assert len(model.grad_history) == 100

    def test_fit_predict_vc_kernels(self):
        prev_mll = -np.inf
        r2_bounds = [0.1, 0.5, 0.7, 0.7]
        for kernel, r2_bound in zip(self.vc_kernels, r2_bounds):
            for use_keops in [False, True]:
                # Infer hyperparameters
                model = EpiK(kernel(self.alpha, self.l, use_keops=use_keops))
                model.set_data(self.X_train, self.y_train, self.y_var)
                model.fit(n_iter=200)

                if not use_keops:
                    assert model.mll >= prev_mll
                else:
                    assert np.allclose(model.mll, prev_mll, atol=5)
                prev_mll = model.mll

                # Predict phenotypes in test data
                test_y_pred = model.predict(self.X_test)["coef"]
                r2 = pearsonr(test_y_pred, self.y_test)[0] ** 2
                assert r2 > r2_bound

    def test_fit_predict_product_kernels(self):
        kernel = GeneralProductKernel(self.alpha, self.l)
        model = EpiK(kernel)
        data = model.simulate_dataset(self.X, sigma=0.1, ptrain=0.9)
        X_train, y_train, test_x, test_y, y_train_var = data

        prev_mll = -np.inf
        for kernel in self.product_kernels:
            for use_keops in [False, True]:
                # Infer hyperparameters
                k = kernel(self.alpha, self.l, use_keops=use_keops)
                model = EpiK(k, track_progress=True, method='cg',
                             n_lanczos_iter=100)
                model.set_data(X_train, y_train, y_train_var)
                model.fit(n_iter=200)

                # Check if the marginal log-likelihood is increasing
                # with more complex models and matches with KeOps
                if not use_keops:
                    assert model.mll >= prev_mll
                else:
                    assert np.allclose(model.mll, prev_mll, atol=10)
                prev_mll = model.mll

                # Predict phenotypes in test data
                test_y_pred = model.predict(test_x)["coef"]
                r2 = pearsonr(test_y_pred, test_y)[0] ** 2
                assert r2 > 0.5

    def test_bin(self):
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

    def test_FA(self):
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
