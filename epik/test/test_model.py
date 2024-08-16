#!/usr/bin/env python
import unittest
import sys

import torch
import pandas as pd
import numpy as np

from os.path import join
from subprocess import check_call
from tempfile import NamedTemporaryFile

from scipy.stats import pearsonr
from gpytorch.kernels import ScaleKernel

from epik.src.settings import TEST_DATA_DIR, BIN_DIR
from epik.src.utils import (seq_to_one_hot, get_tensor, split_training_test,
                            get_full_space_one_hot, one_hot_to_seq,
                            get_mut_effs_contrast_matrix)
from epik.src.model import EpiK
from epik.src.kernel import (VarianceComponentKernel, AdditiveKernel, PairwiseKernel,
                             ConnectednessKernel, ExponentialKernel, AddRhoPiKernel)


def get_smn1_data(n, seed=0, dtype=None):
    np.random.seed(seed)
    data = pd.read_csv(join(TEST_DATA_DIR, 'smn1data.csv'),
                           header=None, index_col=0, names=['m', 'std'])
    data['var'] = data['std'].values ** 2
    data = data.loc[[x[3] == 'U' for x in data.index], :]
    data.index = [x[:3] + x[4:] for x in data.index]
    
    alleles = ['A', 'C', 'G', 'U']
    X, y = seq_to_one_hot(data.index.values, alleles=alleles), data['m'].values
    y_var = data['var']
    
    ps = np.random.uniform(size=data.shape[0])
    p = n / data.shape[0]
    
    train = ps < p
    train_x, train_y = X[train, :], y[train]
    train_y_var = y_var[train] 

    test = ps > (1 - p)
    test_x, test_y = X[test, :], y[test]
    
    ps = np.random.uniform(size=test_x.shape[0])
    p = 1000 / ps.shape[0]
    test_x, test_y = test_x[ps<p], test_y[ps<p]
    
    output = [train_x, train_y, test_x, test_y, train_y_var]
    if dtype is not None:
        output = [get_tensor(a, dtype=dtype) for a in output]
    return(output)


def get_vc_random_landscape_data(sigma=0, ptrain=0.8):
    log_lambdas0 = torch.tensor([-5, 2., 1, 0, -2, -3])
    alpha, l = 4, log_lambdas0.shape[0] - 1
    X = get_full_space_one_hot(seq_length=l, n_alleles=alpha)
    
    kernel = VarianceComponentKernel(n_alleles=alpha, seq_length=l,
                                     log_lambdas0=log_lambdas0)
    model = EpiK(kernel)
    return(alpha, l, log_lambdas0, model.simulate_dataset(X, sigma=sigma, ptrain=ptrain))


def get_additive_random_landscape_data(sigma=0, ptrain=0.8):
    log_lambdas0 = torch.tensor([-10, 0., -10, -10, -10, -10])
    alpha, l = 4, 5
    X = get_full_space_one_hot(seq_length=l, n_alleles=alpha)
    kernel = VarianceComponentKernel(n_alleles=alpha, seq_length=l,
                                     log_lambdas0=log_lambdas0)
    model = EpiK(kernel)
    return(alpha, l, log_lambdas0, model.simulate_dataset(X, sigma=sigma, ptrain=ptrain))


def get_rho_random_landscape_data(sigma=0, ptrain=0.8):
    rho0 = torch.tensor([[0.1, 0.5, 0.05, 0.25, 0.7]]).T
    logit_rho0 = torch.log(rho0 / (1 - rho0))
    alpha, l = 4, logit_rho0.shape[0]
    X = get_full_space_one_hot(seq_length=l, n_alleles=alpha)
    
    kernel = ConnectednessKernel(n_alleles=alpha, seq_length=l, logit_rho0=logit_rho0)
    model = EpiK(kernel)
    return(alpha, l, logit_rho0,
           model.simulate_dataset(X, sigma=sigma, ptrain=ptrain))


class ModelsTests(unittest.TestCase):
    def test_epik_simulate(self):
        l, a = 2, 2
        X = get_full_space_one_hot(seq_length=l, n_alleles=a)
        lambdas0 = torch.tensor([0.001, 1, 0.2])
        prior = LambdasFlatPrior(l, torch.log(lambdas0))
        kernel = VarianceComponentKernel(n_alleles=a, seq_length=l,
                                         lambdas_prior=prior)
        model = EpiK(kernel)
        y = pd.DataFrame(model.sample(X, n=10000).numpy())
        cors = y.corr().values
        
        rho1 = np.array([cors[0, 1], cors[0, 2], cors[1, 0], cors[1, 3],
                         cors[2, 0], cors[2, 3], cors[3, 1], cors[3, 2]])
        rho2 = np.array([cors[0, 3], cors[1, 2], cors[2, 1], cors[3, 0]])
        assert(rho1.std() < 0.2)
        assert(rho2.std() < 0.1)
        
    def test_epik_fit(self):
        alpha, l, log_lambdas0, data = get_vc_random_landscape_data(sigma=0.01)
        train_x, train_y, _, _, train_y_var = data

        # Train new model
        kernel = VarianceComponentKernel(n_alleles=alpha, seq_length=l,
                                         log_lambdas0=log_lambdas0)
        model = EpiK(kernel, track_progress=True)
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=100)
        log_lambdas = kernel.log_lambdas.detach().cpu().numpy().flatten()
        r = pearsonr(log_lambdas[1:], log_lambdas0[1:])[0]
        assert(r > 0.6)

    def test_epik_fit_additive(self):
        alpha, l, log_lambdas0, data = get_vc_random_landscape_data(sigma=0.01)
        train_x, train_y, _, _, train_y_var = data

        # Train new model
        kernel = AdditiveKernel(n_alleles=alpha, seq_length=l)
        model = EpiK(kernel, track_progress=True)
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=100)

    def test_epik_fit_pairwise(self):
        alpha, l, log_lambdas0, data = get_vc_random_landscape_data(sigma=0.01)
        train_x, train_y, _, _, train_y_var = data

        # Train new model
        kernel = PairwiseKernel(n_alleles=alpha, seq_length=l, use_keops=True)
        model = EpiK(kernel, track_progress=True)
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=100)
        log_lambdas = kernel.log_lambdas.detach().cpu().numpy().flatten()
        r = pearsonr(log_lambdas[1:3 + 1], log_lambdas0[1:3])[0]
        assert(r > 0.6)
         
    def test_epik_product(self):
        alpha, l, log_lambdas0, data = get_vc_random_landscape_data(sigma=0.01)
        train_x, train_y, _, _, train_y_var = data
        
        kernel = AdditiveKernel(n_alleles=alpha, seq_length=l) * ConnectednessKernel(n_alleles=alpha, seq_length=l)
        model = EpiK(kernel, track_progress=True)
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=50)
    
    def test_epik_fit_addrho(self):
        alpha, l, log_lambdas0, data = get_vc_random_landscape_data(sigma=0.01)
        train_x, train_y, _, _, train_y_var = data
        
        kernel = AddRhoPiKernel(n_alleles=alpha, seq_length=l)
        model = EpiK(kernel, track_progress=True)
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=50)
        
    def test_epik_fit_addrho_smn1(self):
        l, alpha = 7, 4
        train_x, train_y, test_x, test_y, train_y_var = get_smn1_data(n=1500)
        
        kernel = AddRhoPiKernel(n_alleles=alpha, seq_length=l)
        # kernel = ConnectednessKernel(n_alleles=alpha, seq_length=l) * AdditiveKernel(n_alleles=alpha, seq_length=l)
        model = EpiK(kernel, track_progress=True, learning_rate=0.1)
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=100)
        ypred = model.predict(test_x).detach()
        r2_1 = pearsonr(ypred, test_y)[0] ** 2
        print(r2_1)
        
    def test_epik_fit_rbf(self):
        alpha, l, log_lambdas0, data = get_vc_random_landscape_data(sigma=0.01)
        train_x, train_y, _, _, train_y_var = data
        
        kernel = ExponentialKernel(n_alleles=alpha, seq_length=l)
        model = EpiK(kernel, track_progress=True)
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=50)
        
    def test_epik_predict(self):
        # Simulate from prior distribution
        alpha, l, log_lambdas0, data = get_vc_random_landscape_data(sigma=0, ptrain=0.9)
        train_x, train_y, test_x, test_y, train_y_var = data
        
        # Predict unobserved sequences
        model = EpiK(VarianceComponentKernel(n_alleles=alpha, seq_length=l,
                                             log_lambdas0=log_lambdas0))
        model.set_data(train_x, train_y, train_y_var)
        ypred = model.predict(test_x).detach()
        r2 = pearsonr(ypred, test_y)[0] ** 2
        assert(r2 > 0.9)
        
        y_pred, y_pred_var = model.predict(test_x, calc_variance=True)
        r2 = pearsonr(y_pred.detach(), test_y)[0] ** 2
        assert(r2 > 0.9)
        assert(np.all(y_pred_var.detach().numpy() > 1))

        # With Connectedness model
        alpha, l, logit_rho0, data = get_rho_random_landscape_data(sigma=0, ptrain=0.9)
        train_x, train_y, test_x, test_y, train_y_var = data
        kernel = ConnectednessKernel(alpha, l, logit_rho0=logit_rho0)
        model = EpiK(kernel, track_progress=True)
        model.set_data(train_x, train_y, train_y_var)
        y_pred, _ = model.predict(test_x, calc_variance=True)
        r2 = pearsonr(y_pred, test_y)[0] ** 2
        assert(r2 > 0.9)
        
    def test_epik_contrast(self):
        # Simulate from prior distribution
        alpha, l, log_lambdas0, data = get_vc_random_landscape_data(sigma=0, ptrain=0.9)
        train_x, train_y, test_x, test_y, train_y_var = data
        
        # Define target sequences and contrast
        test_x = torch.tensor([[1, 0, 0, 0, 1, 0, 0, 0] + [1, 0, 0, 0] * 3,
                               [0, 1, 0, 0, 1, 0, 0, 0] + [1, 0, 0, 0] * 3,
                               [1, 0, 0, 0, 0, 1, 0, 0] + [1, 0, 0, 0] * 3,
                               [0, 1, 0, 0, 0, 1, 0, 0] + [1, 0, 0, 0] * 3,])
        contrast_matrix = torch.tensor([[1, -1, -1, 1]])
        
        # Make contrast
        model = EpiK(VarianceComponentKernel(n_alleles=alpha, seq_length=l,
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

    def test_epik_keops(self):
        # Simulate from prior distribution
        alpha, l, logit_rho0, data = get_rho_random_landscape_data(sigma=0, ptrain=0.9)
        train_x, train_y, test_x, test_y, train_y_var = data
        
        # Train new model
        kernel = ConnectednessKernel(alpha, l)
        model = EpiK(kernel, track_progress=True)
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=50)
        logit_rho = kernel.logit_rho.detach().numpy().flatten() 
        r = pearsonr(logit_rho, logit_rho0.flatten())[0]
        assert(r > 0.8)
        
        # Predict unobserved sequences
        kernel = ConnectednessKernel(alpha, l, logit_rho0=logit_rho0)
        model = EpiK(kernel)
        model.set_data(train_x, train_y, train_y_var)
        y_pred = model.predict(test_x)
        r2 = pearsonr(y_pred, test_y)[0] ** 2
        assert(r2 > 0.9)
    
    def test_epik_fit_keops_additive(self):
        alpha, l, _, data = get_additive_random_landscape_data(sigma=0.01)
        train_x, train_y, _, _, train_y_var = data
        
        kernel = AdditiveKernel(n_alleles=alpha, seq_length=l)
        model = EpiK(kernel, track_progress=False)
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=100)
        loglambdas = kernel.log_lambdas.detach().numpy()
        assert(loglambdas[0] < loglambdas[1] - 3.)
    
    def test_epik_keops_gpu2(self):
        # Simulate from prior distribution
        alpha, l, logit_rho0, data = get_rho_random_landscape_data(sigma=0, ptrain=0.9)
        train_x, train_y, test_x, test_y, train_y_var = data
        
        # Train new model
        kernel = ConnectednessKernel(alpha, l)
        model = EpiK(kernel, track_progress=True, device=torch.device('cuda:0'))
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=50)
        logit_rho = kernel.logit_rho.cpu().detach().numpy().flatten() 
        r = pearsonr(logit_rho, logit_rho0.flatten())[0]
        assert(r > 0.8)
        
        # Predict unobserved sequences
        kernel = ConnectednessKernel(alpha, l, logit_rho0=logit_rho0)
        model = EpiK(kernel)
        model.set_data(train_x, train_y, train_y_var)
        ypred = model.predict(test_x).detach()
        r2 = pearsonr(ypred, test_y)[0] ** 2
        assert(r2 > 0.9)
        
    def test_epik_keops_gpu(self):
        np.random.seed(1)
        alpha, l = 2, 100
        n = 2000
        ptrain = 0.5
        
        fpath = '/home/martigo/elzar/programs/epik_analysis/BBQ_data_for_VC_regression/23C.large_effect_loci.0.train.csv'
        data = pd.read_csv(fpath, index_col=0).dropna()
        positions = np.arange(len(data.index.values[0]))
        positions = np.random.choice(positions, size=l, replace=False)
        data['X'] = [''.join(x[p] for p in positions) for x in data.index.values]
        data = data.loc[np.random.choice(data.index.values, size=n, replace=False), :]
        X, y, y_var = seq_to_one_hot(data.X.values, alleles=['A', 'B']), data.y.values, data.y_var.values
        train_x, train_y, test_x, test_y, train_y_var = split_training_test(X, y, y_var=y_var, ptrain=ptrain)

        kernel = AdditiveKernel(alpha, l)
        # kernel = ExponentialKernel(alpha, l)
        # kernel = ConnectednessKernel(alpha, l)
        # kernel = ARDKernel(alpha, l)
        # kernel = ScaleKernel(GPExponentialKernel())
        # kernel = ScaleKernel(GPExponentialKernel(ard_num_dims=train_x.shape[1]))

        model = EpiK(kernel, track_progress=True,
                     device=torch.device('cuda:0'),
                     learning_rate=1.,
                    #  dtype=torch.float64,
                     train_noise=True)
        model.set_data(train_x, train_y, train_y_var + 1e-3)
        model.fit(n_iter=100)
        
        ypred = model.predict(test_x.to(dtype=torch.float64)).detach().cpu().numpy()
        r2 = pearsonr(test_y, ypred)[0] ** 2
        print(ypred.shape, r2)
        print('R2 = {:.2f}'.format(r2))
        
        # print(kernel.base_kernel.raw_lengthscale, kernel.raw_outputscale)
        # print(kernel.base_kernel.lengthscale)
        # exit()
        # logit_rho = kernel.logit_rho.cpu().detach().numpy().flatten() 
        # print(logit_rho)
    
    def test_epik_gpu(self):
        # Simulate from prior distribution
        alpha, l, log_lambdas0, data = get_vc_random_landscape_data(sigma=0, ptrain=0.9)
        train_x, train_y, test_x, test_y, train_y_var = data
        
        # Train new model
        model = EpiK(VarianceComponentKernel(n_alleles=alpha, seq_length=l), 
                     device=torch.device('cuda:0'))
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=100)
        params = model.get_params()
        loglambdas = np.log(params['lambdas'])
        r = pearsonr(loglambdas, log_lambdas0)[0]
        assert(params['mean'][0] == 0)
        assert(r > 0.8)
        
        # Predict unobserved sequences
        kernel = VarianceComponentKernel(n_alleles=alpha, seq_length=l,
                                         lambdas_prior=LambdasFlatPrior(l, log_lambdas0))
        model = EpiK(kernel)
        model.set_data(train_x, train_y, train_y_var)
        ypred = model.predict(test_x).detach()
        r2 = pearsonr(ypred, test_y)[0] ** 2
        assert(r2 > 0.9)
        
    def test_epik_bin(self):
        alleles = np.array(['A', 'C', 'G', 'T'])
        _, _, log_lambdas0, data = get_vc_random_landscape_data(sigma=0, ptrain=0.9)
        train_x, train_y, test_x, test_y, _ = data
        
        data = pd.DataFrame({'y': train_y.numpy()},
                            index=one_hot_to_seq(train_x.numpy(), alleles))
        test = pd.DataFrame({'x': one_hot_to_seq(test_x.numpy(), alleles)})
        bin_fpath = join(BIN_DIR, 'EpiK.py')
        
        with NamedTemporaryFile() as fhand:
            out_fpath = fhand.name
            params_fpath = '{}.model_params.pth'.format(out_fpath)
            data_fpath = '{}.train.csv'.format(out_fpath)
            xpred_fpath = '{}.test.csv'.format(out_fpath)
            data.to_csv(data_fpath)
            test.to_csv(xpred_fpath, header=False, index=False)
        
            # Check help
            cmd = [sys.executable, bin_fpath, '-h']
            check_call(cmd)
            
            # Fit hyperparameters
            cmd = [sys.executable, bin_fpath, data_fpath, '-o', out_fpath, '-n', '50']
            check_call(cmd)
            state_dict = torch.load(params_fpath)
            log_lambdas = state_dict['covar_module.log_lambdas'].numpy().flatten()
            r = pearsonr(log_lambdas, log_lambdas0)[0]
            # assert(r > 0.8)
            
            # Predict test sequences
            cmd = [sys.executable, bin_fpath, data_fpath, '-o', out_fpath, '-n', '0',
                   '-p', xpred_fpath, '--params', params_fpath, '--calc_variance']
            check_call(cmd)
            pred = pd.read_csv(out_fpath, index_col=0)
            print(pred)
            r2 = pearsonr(pred['y_pred'].values, test_y)[0] ** 2
            assert(r2 > 0.8)
            
            ## Test running with different kernel
            # cmd = [sys.executable, bin_fpath, data_fpath, '-o', out_fpath,
            #        '-n', '100', '--gpu', '-p', xpred_fpath,
            #        '-k', 'Rho']
            # check_call(cmd)
    
    def test_epik_bin_gpu(self):
        alleles = np.array(['A', 'C', 'G', 'T'])
        _, _, log_lambdas0, data = get_vc_random_landscape_data(sigma=0, ptrain=0.9)
        train_x, train_y, test_x, test_y, _ = data
        
        data = pd.DataFrame({'y': train_y.numpy()},
                            index=one_hot_to_seq(train_x.numpy(), alleles))
        test = pd.DataFrame({'x': one_hot_to_seq(test_x.numpy(), alleles)})
        bin_fpath = join(BIN_DIR, 'EpiK.py')
        
        with NamedTemporaryFile() as fhand:
            out_fpath = fhand.name
            params_fpath = '{}.model_params.pth'.format(out_fpath)
            data_fpath = '{}.train.csv'.format(out_fpath)
            xpred_fpath = '{}.test.csv'.format(out_fpath)
            data.to_csv(data_fpath)
            test.to_csv(xpred_fpath, header=False, index=False)
        
            # Fitting and prediction using GPU
            cmd = [sys.executable, bin_fpath, data_fpath, '-o', out_fpath,
                   '-n', '100', '--gpu', '-p', xpred_fpath]
            check_call(cmd)
            
            # Check parameter inference
            state_dict = torch.load(params_fpath)
            log_lambdas = state_dict['covar_module.module.raw_theta'].cpu().numpy()
            r = pearsonr(log_lambdas, log_lambdas0)[0]
            assert(r > 0.8)
            
            # Check predictions
            ypred = pd.read_csv(out_fpath, index_col=0)['y_pred'].values
            r2 = pearsonr(ypred, test_y)[0] ** 2
            assert(r2 > 0.9)
    
    def test_bin_long_seqs(self):
        bin_fpath = join(BIN_DIR, 'EpiK.py')
        fpath = '/home/martigo/elzar/programs/epik_analysis/BBQ_data_for_VC_regression/train.csv'
        
        with NamedTemporaryFile() as fhand:
            out_fpath = fhand.name
            cmd = [sys.executable, bin_fpath, fpath, '-o', out_fpath,
                   '-n', '50', '-k', 'Rho']
            check_call(cmd)
            
            params_fpath = '{}.model_params.pth'.format(out_fpath)
            state_dict = torch.load(params_fpath)
            print(state_dict)

    def test_epik_general_product_kernel(self):
        l, alpha = 7, 4
        train_x, train_y, test_x, test_y, train_y_var = get_smn1_data(n=2000)
        track_progress = True
        
        kernel = ScaleKernel(GeneralProductKernel(alpha, l))
        model = EpiK(kernel, track_progress=track_progress, learning_rate=0.1)
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=200)
        ypred = model.predict(test_x).detach()
        r2_1 = pearsonr(ypred, test_y)[0] ** 2
        print(r2_1)
    
    def test_epik_het(self):
        train_x, train_y, test_x, test_y, train_y_var = get_smn1_data(n=1000)
        l, alpha = 7, 4
        track_progress = False
        
        kernel = ScaleKernel(ExponentialKernel(ard_num_dims=1))
        model = EpiK(kernel, track_progress=track_progress, learning_rate=0.2)
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=200)
        ypred = model.predict(test_x).detach()
        r2_1 = pearsonr(ypred, test_y)[0] ** 2
        print(r2_1)
        
        kernel = HetExponentialKernel(alpha, l, dims=1, train_het=False)
        model = EpiK(kernel, track_progress=track_progress)
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=100)
        ypred = model.predict(test_x).detach()
        r2_2 = pearsonr(ypred, test_y)[0] ** 2
        print(r2_2)
        print(kernel.theta)
        print(kernel.theta0)
        
        kernel = HetExponentialKernel(alpha, l, dims=1)
        model = EpiK(kernel, track_progress=track_progress)
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=100)
        ypred = model.predict(test_x).detach()
        r2_2 = pearsonr(ypred, test_y)[0] ** 2
        print(r2_2)
        print(kernel.theta)
        print(kernel.theta0)
        # # exit()
        kernel = ScaleKernel(ExponentialKernel(ard_num_dims=alpha * l))
        model = EpiK(kernel, track_progress=track_progress, learning_rate=0.2)
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=200)
        ypred = model.predict(test_x).detach()
        r2_3 = pearsonr(ypred, test_y)[0] ** 2
        print(r2_3)
        
        kernel = HetExponentialKernel(alpha, l, dims=alpha*l)
        model = EpiK(kernel, track_progress=track_progress)
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=15)
        ypred = model.predict(test_x).detach()
        r2_4 = pearsonr(ypred, test_y)[0] ** 2
        print(r2_4)
        
        print(kernel.theta)
        print(kernel.theta0)
        
        print(r2_1, r2_2, r2_3, r2_4)
            
    def test_partitioning(self):
        # Simulate from prior distribution
        alpha, l, log_lambdas0, data = get_vc_random_landscape_data(sigma=0, ptrain=0.9)
        train_x, train_y, test_x, test_y, train_y_var = data
        
        # Fit model
        model = EpiK(VarianceComponentKernel(n_alleles=alpha, seq_length=l),
                     partition_size=1e5)
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=100)
        params = model.get_params()
        loglambdas = np.log(params['lambdas'])
        r = pearsonr(loglambdas, log_lambdas0)[0]
        assert(params['mean'][0] == 0)
        assert(r > 0.8)
        
        # Predict unobserved sequences
        ypred = model.predict(test_x).detach()
        r2 = pearsonr(ypred, test_y)[0] ** 2
        assert(r2 > 0.9)

        
if __name__ == '__main__':
    import sys; sys.argv = ['', 'ModelsTests.test_epik_keops_gpu']
    unittest.main()
