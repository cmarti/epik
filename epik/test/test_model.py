#!/usr/bin/env python
import sys
import gc
import unittest
from functools import partial
from os.path import join
from subprocess import check_call
from tempfile import NamedTemporaryFile

import gpytorch
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, multivariate_normal

from epik.src.kernel import (
    AdditiveKernel,
    ConnectednessKernel,
    ExponentialKernel,
    GeneralProductKernel,
    JengaKernel,
    PairwiseKernel,
    VarianceComponentKernel,
)
from epik.src.model import EpiK
from epik.src.settings import BIN_DIR
from epik.src.utils import (
    get_full_space_one_hot,
    get_mut_effs_contrast_matrix,
    one_hot_to_seq,
    seq_to_one_hot,
    encode_seqs,
    get_random_sequences,
    calc_distance_covariance
)


def get_vc_random_landscape_data(sigma=0, ptrain=0.8):
    log_lambdas0 = torch.tensor([-5, 2., 1, 0, -2, -3])
    alpha, sl = 4, log_lambdas0.shape[0] - 1
    X = get_full_space_one_hot(seq_length=sl, n_alleles=alpha)
    
    kernel = VarianceComponentKernel(n_alleles=alpha, seq_length=sl,
                                     log_lambdas0=log_lambdas0)
    model = EpiK(kernel)
    return(alpha, sl, log_lambdas0, model.simulate_dataset(X, sigma=sigma, ptrain=ptrain))


class ModelsTests(unittest.TestCase):
    def test_calc_mll(self):
        alphabet = ['A', 'C', 'G', 'T']
        n_alleles= len(alphabet)
        seq_length = 8
        for n in [100, 200, 500, 1000]:
            seqs = get_random_sequences(n=n, seq_length=seq_length, alphabet=alphabet)
            X = encode_seqs(seqs, alphabet=alphabet)
            y_var = 0.1 * np.ones(n)
            D = np.diag(y_var)
            with torch.no_grad():
                kernel = ConnectednessKernel(n_alleles=n_alleles, seq_length=seq_length)
                mu = np.zeros(n)
                Sigma = kernel(X, X).to_dense().numpy() + D
                
                gaussian = multivariate_normal(mu, Sigma)
                y = gaussian.rvs()
                logp1 = gaussian.logpdf(y)
                
                model = EpiK(kernel)
                model.set_data(X=X, y=y, y_var=y_var)
                logp2 = model.calc_mll()
                
                print(logp1, logp2)
                assert(np.allclose(logp1, logp2))
        
    def test_simulate(self):
        sl, a = 2, 2
        X = get_full_space_one_hot(seq_length=sl, n_alleles=a)
        lambdas0 = torch.log(torch.tensor([0.001, 1, 0.2]))
        kernel = VarianceComponentKernel(n_alleles=a, seq_length=sl,
                                         log_lambdas0=lambdas0)
        model = EpiK(kernel)
        y = pd.DataFrame(model.simulate(X, n=10000).numpy())
        cors = y.corr().values
        
        rho1 = np.array([cors[0, 1], cors[0, 2], cors[1, 0], cors[1, 3],
                         cors[2, 0], cors[2, 3], cors[3, 1], cors[3, 2]])
        rho2 = np.array([cors[0, 3], cors[1, 2], cors[2, 1], cors[3, 0]])
        assert(rho1.std() < 0.2)
        assert(rho2.std() < 0.1)
    
    def test_predict(self):
        # Simulate from prior distribution
        alpha, sl, log_lambdas0, data = get_vc_random_landscape_data(sigma=0, ptrain=0.9)
        train_x, train_y, test_x, test_y, train_y_var = data
        
        # Set up model and data
        kernel = VarianceComponentKernel(n_alleles=alpha, seq_length=sl,
                                         log_lambdas0=log_lambdas0)
        model = EpiK(kernel)
        model.set_data(train_x, train_y, train_y_var)

        # Predict unobserved sequences
        results = model.predict(test_x, calc_variance=False)
        r2 = pearsonr(results['coef'], test_y)[0] ** 2
        assert(r2 > 0.9)
        
        results = model.predict(test_x, calc_variance=True)
        r2 = pearsonr(results['coef'], test_y)[0] ** 2
        assert(r2 > 0.9)

    def test_contrast(self):
        # Simulate from prior distribution
        alpha, sl, log_lambdas0, data = get_vc_random_landscape_data(sigma=0, ptrain=0.9)
        train_x, train_y, test_x, test_y, train_y_var = data
        
        # Define target sequences and contrast
        test_x = torch.tensor([[1, 0, 0, 0, 1, 0, 0, 0] + [1, 0, 0, 0] * 3,
                               [0, 1, 0, 0, 1, 0, 0, 0] + [1, 0, 0, 0] * 3,
                               [1, 0, 0, 0, 0, 1, 0, 0] + [1, 0, 0, 0] * 3,
                               [0, 1, 0, 0, 0, 1, 0, 0] + [1, 0, 0, 0] * 3,])
        contrast_matrix = torch.tensor([[1, -1, -1, 1]])
        
        # Make contrast
        model = EpiK(VarianceComponentKernel(n_alleles=alpha, seq_length=sl,
                                             log_lambdas0=log_lambdas0))
        model.set_data(train_x, train_y, train_y_var)
        m, cov = model.make_contrasts(contrast_matrix, test_x, calc_variance=True)
        assert(m.shape == (1,))
        assert(cov.shape == (1, 1))
        
        # Make contrast with built-in matrix functions
        contrast_matrix = get_mut_effs_contrast_matrix(seq0='AAAAA', alleles='ACGT')
        test_x = seq_to_one_hot(contrast_matrix.columns, alleles='ACGT')
        contrast_matrix = torch.Tensor(contrast_matrix.values)
        m, cov = model.make_contrasts(contrast_matrix, test_x, calc_variance=True)
        assert(m.shape == (15,))
        assert(cov.shape == (15, 15))

        # Make contrast with built-in method
        results = model.predict_mut_effects(seq0='AAAAA', alleles='ACGT', calc_variance=True)
        assert(results.shape == (15, 4))
        
        results = model.predict_mut_effects(seq0='AAAAA', alleles='ACGT', calc_variance=False)
        assert(results.shape == (15, 1))
        
    def test_fit(self):
        torch.manual_seed(0)
        alpha, sl, log_lambdas0, data = get_vc_random_landscape_data(sigma=0.01)
        train_x, train_y, _, _, train_y_var = data

        # Train new model
        kernel = VarianceComponentKernel(n_alleles=alpha, seq_length=sl)
        model = EpiK(kernel, track_progress=True)
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=100, learning_rate=0.01)
        log_lambdas = kernel.log_lambdas.detach().cpu().numpy().flatten()
        r = pearsonr(log_lambdas[1:], log_lambdas0[1:])[0]
        assert(r > 0.6)
    
    def test_fit2(self):
        torch.manual_seed(2)
        
        # data = pd.read_csv('/home/martigo/elzar/projects/epik_analysis/splits/qtls_li_hq.24.train.csv', index_col=0)
        data = pd.read_csv('/home/martigo/elzar/projects/epik_analysis/splits/smn1.24.train.csv', index_col=0)
        print(data)
        X = encode_seqs(data.index.values, alphabet=['A', 'C', 'G', 'U'])
        y = torch.Tensor(data['y'])
        y_var = torch.Tensor(data['y_var'])
        
        # Train new model
        # kernel = GeneralProductKernel(n_alleles=20, seq_length=4)
        kernel = JengaKernel(n_alleles=4, seq_length=8)
        model = EpiK(kernel,
                    track_progress=True,
                    train_noise=True)
        model.set_data(X, y, y_var)
        
        # for n in [1, 2, 4, 8, 16, 32, 64, 128]:
        #     with gpytorch.settings.cg_tolerance(1 ), gpytorch.settings.num_trace_samples(n):
        #         values = []
        #         for i in range(20):
        #             values.append(model.calc_mll().detach().numpy())
        #         print(np.mean(values), np.std(values))
        # # exit
        model.fit(n_iter=50, learning_rate=1e-4)
        model.fit(n_iter=100, learning_rate=1e-6)
        print(model.training_history)
        print(model.max_mll)
    
    def test_fit_predict_kernels(self):
        torch.manual_seed(0)
        log_lambdas = torch.tensor([0, 1, 0, -5, -10])
        seq_length, n_alleles = log_lambdas.shape[0] - 1, 4
        X = get_full_space_one_hot(seq_length=seq_length, n_alleles=n_alleles)
    
        for use_keops in [False, True]:
            for kernel in [
                partial(AdditiveKernel, log_lambdas0=log_lambdas[:2]),
                partial(PairwiseKernel, log_lambdas0=log_lambdas[:3]),
                partial(VarianceComponentKernel, log_lambdas0=log_lambdas),
                ExponentialKernel,
                ConnectednessKernel,
                JengaKernel,
                GeneralProductKernel,
            ]:
                # Simulate data
                model = EpiK(kernel(n_alleles, seq_length, use_keops=use_keops))
                data = model.simulate_dataset(X, sigma=0.1, ptrain=0.9)
                train_x, train_y, test_x, test_y, train_y_var = data

                # Infer hyperparameters
                # cov0, ns0 = calc_distance_covariance(train_x, train_y, seq_length)
                K = kernel(n_alleles, seq_length, use_keops=use_keops)
                model = EpiK(K, track_progress=True)
                model.set_data(train_x, train_y, train_y_var)
                model.fit(n_iter=100)

                # Predict phenotypes in test data
                test_y_pred = model.predict(test_x)['coef']
                r2 = pearsonr(test_y_pred, test_y)[0] ** 2
                assert(r2 > 0.5)
    
    def test_fit_predict_kernels_bin(self):
        log_lambdas = torch.tensor([0, 1, 0, -5, -10])
        seq_length = log_lambdas.shape[0] - 1
        alleles = np.array(['A', 'C', 'G', 'T'])
        ref_seq = 'A' * seq_length
        n_alleles = len(alleles)
        X = get_full_space_one_hot(seq_length=seq_length, n_alleles=n_alleles)
        bin_fpath = join(BIN_DIR, "EpiK.py")
    
        labels = ['Additive',
                  'Pairwise', 'VC', 'Exponential',
                  'Connectedness', 'Jenga', 'GeneralProduct']
        for kernel, label in zip([
            partial(AdditiveKernel, log_lambdas0=log_lambdas[:2]),
            partial(PairwiseKernel, log_lambdas0=log_lambdas[:3]),
            partial(VarianceComponentKernel, log_lambdas0=log_lambdas),
            ExponentialKernel,
            ConnectednessKernel,
            JengaKernel,
            GeneralProductKernel,
        ], labels):
            
            # Simulate data
            model = EpiK(kernel(n_alleles, seq_length))
            data = model.simulate_dataset(X, sigma=0.1, ptrain=0.9)
            train_x, train_y, test_x, test_y, train_y_var = data
            train_seqs = one_hot_to_seq(train_x.numpy(), alleles)
            test_seqs = one_hot_to_seq(test_x.numpy(), alleles)
            data = pd.DataFrame({'y': train_y.numpy()}, index=train_seqs)
            test = pd.DataFrame({'x': test_seqs})
            
            with NamedTemporaryFile() as fhand:
                out_fpath = fhand.name
                params_fpath = '{}.model_params.pth'.format(out_fpath)
                data_fpath = '{}.train.csv'.format(out_fpath)
                xpred_fpath = '{}.test.csv'.format(out_fpath)
                data.to_csv(data_fpath)
                test.to_csv(xpred_fpath, header=False, index=False)
            
            # Fit hyperparameters
            cmd = [sys.executable, bin_fpath, data_fpath,
                '-k', label, '-o', out_fpath, '-n', '50']
            check_call(cmd)
            
            # Predict test sequences
            cmd = [sys.executable, bin_fpath, data_fpath,
                '-k', label,  '-o', out_fpath, '-n', '0',
                '-p', xpred_fpath, '--params', params_fpath, '--calc_variance']
            check_call(cmd)
            pred = pd.read_csv(out_fpath, index_col=0)
            r2 = pearsonr(pred['coef'].values, test_y)[0] ** 2
            assert(r2 > 0.6)

            # Calculate mutational effects contrasts
            cmd = [sys.executable, bin_fpath, data_fpath,
                '-k', label,  '-o', out_fpath, '-n', '0',
                '-s', ref_seq, '--params', params_fpath, '--calc_variance']
            check_call(cmd)
    
        
if __name__ == '__main__':
    import sys
    sys.argv = ['', 'ModelsTests']
    unittest.main()
