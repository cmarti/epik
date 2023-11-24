#!/usr/bin/env python
import unittest
import sys

import torch
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from os.path import join
from subprocess import check_call
from tempfile import NamedTemporaryFile

from scipy.stats import pearsonr
from gpmap.src.inference import VCregression

from gpytorch.settings import max_cg_iterations
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel

from epik.src.settings import TEST_DATA_DIR, BIN_DIR
from epik.src.utils import (seq_to_one_hot, get_tensor, split_training_test,
                            get_full_space_one_hot, one_hot_to_seq)
from epik.src.model import EpiK
from epik.src.kernel import SkewedVCKernel, VarianceComponentKernel
from epik.src.priors import (LambdasExpDecayPrior, AllelesProbPrior,
                             LambdasDeltaPrior, LambdasFlatPrior, RhosPrior)
from epik.src.plot import plot_training_history
from itertools import product


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
    log_lambdas0 = torch.tensor([-5, 2., 1, 0, -2, -5])
    alpha, l = 4, log_lambdas0.shape[0] - 1
    X = get_full_space_one_hot(seq_length=l, n_alleles=alpha)
    
    kernel = VarianceComponentKernel(n_alleles=alpha, seq_length=l,
                                     lambdas_prior=LambdasFlatPrior(l, log_lambdas0))
    model = EpiK(kernel)
    return(alpha, l, log_lambdas0, model.simulate_dataset(X, sigma=sigma, ptrain=ptrain))


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
        model = EpiK(VarianceComponentKernel(n_alleles=alpha, seq_length=l),
                     optimizer='Adam', track_progress=False)
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=100)
        params = model.get_params()
        loglambdas = np.log(params['lambdas'])
        r = pearsonr(loglambdas, log_lambdas0)[0]
        print(r, loglambdas)
        assert(r > 0.8)
        
        # Train LBFGS optimizer
        model = EpiK(VarianceComponentKernel(n_alleles=alpha, seq_length=l),
                     optimizer='LBFGS', track_progress=False)
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=300)
        params = model.get_params()
        loglambdas = np.log(params['lambdas'])
        r = pearsonr(loglambdas, log_lambdas0)[0]
        assert(params['mean'][0] == 0)
        assert(r > 0.8)
    
    def test_epik_predict(self):
        # Simulate from prior distribution
        alpha, l, log_lambdas0, data = get_vc_random_landscape_data(sigma=0, ptrain=0.9)
        train_x, train_y, test_x, test_y, train_y_var = data
        
        # Predict unobserved sequences
        kernel = VarianceComponentKernel(n_alleles=alpha, seq_length=l,
                                         lambdas_prior=LambdasFlatPrior(l, log_lambdas0))
        model = EpiK(kernel)
        model.set_data(train_x, train_y, train_y_var)
        ypred = model.predict(test_x).detach()
        r2 = pearsonr(ypred, test_y)[0] ** 2
        assert(r2 > 0.9)
    
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
            
            # Model fitting
            cmd = [sys.executable, bin_fpath, data_fpath, '-o', out_fpath,
                   '-n', '100']
            check_call(cmd)
            state_dict = torch.load(params_fpath)
            log_lambdas = state_dict['covar_module.raw_theta'].numpy()
            r = pearsonr(log_lambdas, log_lambdas0)[0]
            assert(r > 0.8)
            
            # Predict test sequences
            cmd.extend(['-p', xpred_fpath, '--params', params_fpath])
            check_call(cmd)
            ypred = pd.read_csv(out_fpath, index_col=0)['y_pred'].values
            r2 = pearsonr(ypred, test_y)[0] ** 2
            assert(r2 > 0.9)
            
            # Test running with different kernel
            cmd = [sys.executable, bin_fpath, data_fpath, '-o', out_fpath,
                   '-n', '100', '--gpu', '-p', xpred_fpath,
                   '-k', 'Rho']
            check_call(cmd)
    
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
    
    def test_epik_rho_kernel(self):
        alleles = ['A', 'C', 'T', 'G']
        rho0 = torch.tensor([0.1, 0.6, 0.7, 0.6, 0.4])
        log_rho0 = np.log(rho0)
        l, a = rho0.shape[0], len(alleles)
        seqs = np.array([''.join(gt) for gt in product(alleles, repeat=l)])
        X = seq_to_one_hot(seqs, alleles)
        sigma2 = 0.1
        y_var = sigma2 * torch.ones(X.shape[0])
        
        # Simulate under rho model
        rhos_prior = RhosPrior(seq_length=l, n_alleles=a, rho0=rho0)
        p_prior = AllelesProbPrior(seq_length=l, n_alleles=a, dummy_allele=False)
        kernel = GeneralizedSiteProductKernel(n_alleles=a, seq_length=l,
                                              rho_prior=rhos_prior, p_prior=p_prior)
        model = EpiK(kernel, likelihood_type='Gaussian')
        y = model.sample(X, n=1, sigma2=sigma2).flatten()
        
        # Split data in training and test sets
        splits = split_training_test(X, y, y_var, ptrain=0.8)
        train_x, train_y, test_x, test_y, train_y_var = splits
        
        # Train new model
        rhos_prior = RhosPrior(seq_length=l, n_alleles=a)
        p_prior = AllelesProbPrior(seq_length=l, n_alleles=a, train=False,
                                   alleles_equal=True, sites_equal=True)
        kernel = GeneralizedSiteProductKernel(n_alleles=a, seq_length=l,
                                              rho_prior=rhos_prior, p_prior=p_prior)
        model = EpiK(kernel, likelihood_type='Gaussian')
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=50)
        
        # Test rho inference
        params = kernel.get_params()
        log_rho = np.log(params['rho'].detach().numpy())
        r = pearsonr(log_rho, log_rho0)[0]
        assert(r > 0.7)
        
        # Predict on test sequences
        ypred = model.predict(test_x).detach()
        r2 = pearsonr(ypred, test_y)[0] ** 2
        assert(r2 > 0.9)
    
    def test_epik_rho_pi_kernel(self):
        alleles = ['A', 'C', 'T', 'G']
        rho0 = torch.tensor([0.1, 0.6, 0.7, 0.6, 0.4])
        rho0 = torch.tensor([0.5,])
        p0 = torch.tensor([[0.2, 0.5, 0.2, 0.1],
                           [0.1, 0.1, 0.5, 0.3],
                           [0.2, 0.5, 0.2, 0.1],
                           [0.2, 0.5, 0.2, 0.1],
                           [0.3, 0.25, 0.2, 0.25],
                           [0.25, 0.25, 0.25, 0.25]])
        p0 = torch.tensor([[0.01, 0.49, 0.49, 0.01]])
        beta0 = -torch.log(p0 / (1-p0))
        log_rho0 = np.log(rho0)
        l, a = rho0.shape[0], len(alleles)
        l = 5
        print(l, a)
        seqs = np.array([''.join(gt) for gt in product(alleles, repeat=l)])
        X = seq_to_one_hot(seqs, alleles)
        sigma2 = 0.005
        y_var = sigma2 * torch.ones(X.shape[0])
        
        # Simulate under rho model
        rhos_prior = RhosPrior(seq_length=l, n_alleles=a, rho0=rho0, sites_equal=True,
                               train=False)
        p_prior = AllelesProbPrior(seq_length=l, n_alleles=a, dummy_allele=False,
                                   beta0=beta0, sites_equal=True)
        assert(np.allclose(p_prior.norm_logp_to_beta(np.log(p0)), beta0))
        
        kernel = GeneralizedSiteProductKernel(n_alleles=a, seq_length=l,
                                              rho_prior=rhos_prior, p_prior=p_prior)
        model = EpiK(kernel, likelihood_type='Gaussian')
        y = model.sample(X, n=1, sigma2=sigma2).flatten()
        
        # Split data in training and test sets
        splits = split_training_test(X, y, y_var, ptrain=0.8)
        train_x, train_y, test_x, test_y, train_y_var = splits
        
        # Train new model
        # rhos_prior = RhosPrior(seq_length=l, n_alleles=a, sites_equal=True)
        p_prior = AllelesProbPrior(seq_length=l, n_alleles=a, train=True,
                                   dummy_allele=False, sites_equal=True)
        kernel = GeneralizedSiteProductKernel(n_alleles=a, seq_length=l,
                                              rho_prior=rhos_prior, p_prior=p_prior)
        model = EpiK(kernel, likelihood_type='Gaussian', train_mean=False,
                     learning_rate=0.1)
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=500)
        
        # Test rho inference
        params = kernel.get_params()
        log_rho = np.log(params['rho'].detach().numpy())
        # r = pearsonr(log_rho, log_rho0)[0]
        # assert(r > 0.7)
        print(log_rho, log_rho0)
        
        # Test p inference
        beta = params['beta'].detach().numpy()[0].flatten()
        print(beta)
        print(beta0)
        r = pearsonr(beta[0].flatten(), beta0.flatten())[0]
        print(r)
        
        # Predict on test sequences
        ypred = model.predict(test_x).detach()
        r2 = pearsonr(ypred, test_y)[0] ** 2
        assert(r2 > 0.9)
        
    def test_epik_vc_smn1(self):
        train_x, train_y, test_x, test_y, train_y_var = get_smn1_data(n=1000)
        l, a = 7, 4
        
        lambdas_prior = LambdasFlatPrior(seq_length=l)
        kernel = VCKernel(n_alleles=a, seq_length=l, lambdas_prior=lambdas_prior)
        model = EpiK(kernel)
        model.fit(train_x, train_y, y_var=train_y_var, n_iter=200, learning_rate=0.02)
        
        train_ypred = model.predict(train_x).detach().numpy()
        test_ypred = model.predict(test_x).detach().numpy()
        
        train_rho = pearsonr(train_ypred, train_y)[0]
        test_rho = pearsonr(test_ypred, test_y)[0]
        print(kernel.lambdas.detach(), test_rho)
        
        assert(train_rho > 0.9)
        assert(test_rho > 0.6)
    
    def test_epik_vc_smn1_gpu(self):
        train_x, train_y, test_x, test_y, train_y_var = get_smn1_data(n=2000)
        output_device = torch.device('cuda:0')
        l, a = 7, 4
        
        lambdas_prior = LambdasFlatPrior(seq_length=l)
        kernel = VCKernel(n_alleles=a, seq_length=l, lambdas_prior=lambdas_prior)
        model = EpiK(kernel, output_device=output_device)
        model.fit(train_x, train_y, y_var=train_y_var, n_iter=100, learning_rate=0.02)
        
        train_ypred = model.predict(train_x).cpu().detach().numpy()
        test_ypred = model.predict(test_x).cpu().detach().numpy()
        
        train_rho = pearsonr(train_ypred, train_y)[0]
        test_rho = pearsonr(test_ypred, test_y)[0]
        assert(train_rho > 0.9)
        assert(test_rho > 0.7)
    
    def test_epik_vc_smn1_exp_decay(self):
        train_x, train_y, test_x, test_y, train_y_var = get_smn1_data(n=1000)
        l, a = 7, 4
        
        lambdas_prior = LambdasExpDecayPrior(seq_length=l)
        kernel = VCKernel(n_alleles=a, seq_length=l, lambdas_prior=lambdas_prior)
        model = EpiK(kernel)
        model.fit(train_x, train_y, y_var=train_y_var, n_iter=200, learning_rate=0.02)
        
        train_ypred = model.predict(train_x).detach().numpy()
        test_ypred = model.predict(test_x).detach().numpy()
        
        train_rho = pearsonr(train_ypred, train_y)[0]
        test_rho = pearsonr(test_ypred, test_y)[0]
        print(kernel.lambdas.detach(), test_rho)
        
        assert(train_rho > 0.9)
        assert(test_rho > 0.6)
    
    def test_epik_delta_smn1(self):
        train_x, train_y, test_x, test_y, train_y_var = get_smn1_data(n=1000)
        l, a = 7, 4
        
        lambdas_prior = LambdasDeltaPrior(seq_length=l, n_alleles=a, P=2)
        kernel = VCKernel(n_alleles=a, seq_length=l, lambdas_prior=lambdas_prior)
        model = EpiK(kernel)
        model.fit(train_x, train_y, y_var=train_y_var, n_iter=100, learning_rate=0.02)
        
        train_ypred = model.predict(train_x).detach().numpy()
        test_ypred = model.predict(test_x).detach().numpy()
        
        train_rho = pearsonr(train_ypred, train_y)[0]
        test_rho = pearsonr(test_ypred, test_y)[0]
        
        assert(train_rho > 0.9)
        assert(test_rho > 0.6)
        
    def test_epik_delta_smn1_gpu(self):
        train_x, train_y, test_x, test_y, train_y_var = get_smn1_data(n=1000)
        output_device = torch.device('cuda:0')
        l, a = 7, 4
        
        lambdas_prior = LambdasDeltaPrior(seq_length=l, n_alleles=a, P=2)
        kernel = VCKernel(n_alleles=a, seq_length=l, lambdas_prior=lambdas_prior)
        model = EpiK(kernel, output_device=output_device)
        model.fit(train_x, train_y, y_var=train_y_var, n_iter=200, learning_rate=0.02)
        
        train_ypred = model.predict(train_x).detach().cpu().numpy()
        test_ypred = model.predict(test_x).detach().cpu().numpy()
        
        train_rho = pearsonr(train_ypred, train_y)[0]
        test_rho = pearsonr(test_ypred, test_y)[0]
        
        assert(train_rho > 0.9)
        assert(test_rho > 0.6)
        
    def test_epik_basic_RBF(self):
        kernel = ScaleKernel(RBFKernel())
        model = EpiK(kernel, likelihood_type='Gaussian')
        
        X = seq_to_one_hot(np.array(['AA', 'AB', 'BA', 'BB']))
        y = torch.tensor([0.2, 1.1, 0.5, 1.5])
        model.fit(X, y, n_iter=200)
        ypred = model.predict(X)
        print(ypred)
    
    def test_rbf_smn1(self):
        train_x, train_y, test_x, test_y, train_y_var = get_smn1_data(n=1000)
        
        kernel = ScaleKernel(RBFKernel())
        model = EpiK(kernel, likelihood_type='Gaussian')
        model.fit(train_x, train_y, y_var=train_y_var,
                  n_iter=100, learning_rate=0.05)
        
        train_ypred = model.predict(train_x)
        test_ypred = model.predict(test_x).detach().numpy()
        
        train_rho = pearsonr(train_ypred, train_y)[0]
        test_rho = pearsonr(test_ypred, test_y)[0]
        assert(train_rho > 0.7)
        assert(test_rho > 0.6)
    
    def test_epik_skewed_vc_smn1(self):
        train_x, train_y, test_x, test_y, train_y_var = get_smn1_data(n=1000)
        l, a = 7, 4
        
        lambdas_prior = LambdasExpDecayPrior(seq_length=l, tau=0.2)
        p_prior = AllelesProbPrior(seq_length=l, n_alleles=a)
        kernel = SkewedVCKernel(n_alleles=a, seq_length=l, q=0.7,
                                lambdas_prior=lambdas_prior, p_prior=p_prior)
        model = EpiK(kernel)
        model.fit(train_x, train_y, y_var=train_y_var,
                  n_iter=100, learning_rate=0.05)
        
        train_ypred = model.predict(train_x).detach().numpy()
        test_ypred = model.predict(test_x).detach().numpy()
        
        train_rho = pearsonr(train_ypred, train_y)[0]
        test_rho = pearsonr(test_ypred, test_y)[0]
        
        assert(train_rho > 0.9)
        assert(test_rho > 0.7)
        
    def test_epik_skewed_vc_smn1_gpu(self):
        train_x, train_y, test_x, test_y, train_y_var = get_smn1_data(n=1000)
        l, a = 7, 4
        n_devices, output_device = 1, torch.device('cuda:0') 
        
        lambdas_prior = LambdasFlatPrior(seq_length=l)
        p_prior = AllelesProbPrior(seq_length=l, n_alleles=a)
        kernel = SkewedVCKernel(n_alleles=a, seq_length=l, q=0.7,
                                lambdas_prior=lambdas_prior, p_prior=p_prior,
                                n_devices=n_devices, output_device=output_device)
        model = EpiK(kernel)
        model.fit(train_x, train_y, y_var=train_y_var,
                  n_iter=100, learning_rate=0.05)
        
        train_ypred = model.predict(train_x).detach().cpu().numpy()
        test_ypred = model.predict(test_x).detach().cpu().numpy()
        
        train_rho = pearsonr(train_ypred, train_y)[0]
        test_rho = pearsonr(test_ypred, test_y)[0]
        
        plot_training_history(model.loss_history, join(TEST_DATA_DIR, 'test_history.png'))
        
        assert(train_rho > 0.9)
        assert(test_rho > 0.7)
        
    def test_epik_site_kernel_smn1(self):
        train_x, train_y, test_x, test_y, train_y_var = get_smn1_data(n=1000)
        l, a = 7, 4
        
        p_prior = AllelesProbPrior(seq_length=l, n_alleles=a, alleles_equal=True)
        kernel = SiteProductKernel(n_alleles=4, seq_length=7, p_prior=p_prior)
        model = EpiK(kernel)
        model.fit(train_x, train_y, y_var=train_y_var,
                  n_iter=100, learning_rate=0.05)
        
        train_ypred = model.predict(train_x).detach().numpy()
        test_ypred = model.predict(test_x).detach().numpy()
        
        train_rho = pearsonr(train_ypred, train_y)[0]
        test_rho = pearsonr(test_ypred, test_y)[0]
        
        w = kernel.beta.detach().numpy()[:, 0]
        assert(w[0] < w[1])
        assert(w[-1] < w[-2])
        assert(train_rho > 0.9)
        assert(test_rho > 0.6)
    
    def test_epik_site_kernel_smn1_2(self):
        seq_length = 220
#         seq_length = 100
        train = pd.read_parquet('/home/cmarti/Downloads/BBQ_data_for_VC_regression/data.train.pq')
        train = train.iloc[:2000, :]
        test = pd.read_parquet('/home/cmarti/Downloads/BBQ_data_for_VC_regression/data.test.pq')
        test = test.iloc[:500, :]
        
        
        x_train, y_train, y_var_train = seq_to_one_hot(train.index, alleles=['A', 'B']), train['li_m'].values, train['li_var'].values
        x_train = x_train[: , :2*seq_length]
        logv0 = np.log(y_train.var())
        print(x_train.shape, logv0)
        x_test, y_test, y_var_test = seq_to_one_hot(test.index, alleles=['A', 'B']), test['li_m'].values, test['li_var'].values
        x_test = x_test[: , :2*seq_length]
        
        rho_prior = RhosPrior(seq_length=seq_length, n_alleles=2, sites_equal=False,
                              v0=1.05)
        p_prior = AllelesProbPrior(seq_length=seq_length, n_alleles=2,
                                   alleles_equal=True, sites_equal=False,
                                   dummy_allele=False, )
        kernel = GeneralizedSiteProductKernel(n_alleles=2, seq_length=seq_length,
                                              p_prior=p_prior, rho_prior=rho_prior)
#         kernel = RBFKernel()
#         kernel.lengthscale = 20
#         kernel = ScaleKernel(kernel)

        model = EpiK(kernel, likelihood_type='Gaussian', learning_rate=0.01)
        model.set_data(x_train, y_train)
        model.fit(n_iter=50)
        
        test_ypred = model.predict(x_test).detach().numpy()
        train_ypred = model.predict(x_train).detach().numpy()
        test_rho = pearsonr(test_ypred, y_test)[0]
        train_rho = pearsonr(train_ypred, y_train)[0]
        print(test_rho, train_rho)
        print(model.get_gp_mean())
        
        w = kernel.beta.detach().numpy()[:, 0]
        assert(w[0] < w[1])
        assert(w[-1] < w[-2])
        assert(test_rho > 0.6)
    
    def test_epik_generalized_site_kernel_smn1(self):
        max_cg_iterations(2000)
        train_x, train_y, test_x, test_y, train_y_var = get_smn1_data(n=2000)
        l, a = 7, 4
        
        rho_prior = RhosPrior(seq_length=l, n_alleles=a,
                              sites_equal=False, v0=1.05)
        p_prior = AllelesProbPrior(seq_length=l, n_alleles=a,
                                   alleles_equal=False, sites_equal=False,
                                   dummy_allele=False)
        kernel = GeneralizedSiteProductKernel(n_alleles=a, seq_length=l,
                                              p_prior=p_prior, rho_prior=rho_prior)
        model = EpiK(kernel, likelihood_type='Gaussian', learning_rate=0.05)
        model.set_data(train_x, train_y, y_var=train_y_var)
        model.fit(n_iter=40)
        
        train_ypred = model.predict(train_x).detach().numpy()
        test_ypred = model.predict(test_x).detach().numpy()
        
        train_rho = pearsonr(train_ypred, train_y)[0]
        test_rho = pearsonr(test_ypred, test_y)[0]
        
        w = kernel.beta.detach().numpy()
        print(w)
        rho = kernel.rho.detach().numpy()
        print(rho)
        print(train_rho**2, test_rho**2)
        
        # assert(w[0] < w[1])
        # assert(w[-1] < w[-2])
        # assert(train_rho > 0.9)
        # assert(test_rho > 0.6)
        
        fig, axes = plt.subplots(1, 1, figsize=(6, 6))
        axes.scatter(test_ypred, test_y)
        fig.savefig('scatter.png', dpi=300)
        
        plot_training_history(model.loss_history,
                              join(TEST_DATA_DIR, 'test_history.png'))
    
    def test_epik_site_kernel_smn1_gpu(self):
        train_x, train_y, test_x, test_y, train_y_var = get_smn1_data(n=1500)
        output_device = torch.device('cuda:0')
        
        kernel = SiteProductKernel(n_alleles=4, seq_length=7)
        model = EpiK(kernel, likelihood_type='Gaussian',
                     output_device=output_device, learning_rate=0.05)
        model.set_data(train_x, train_y, y_var=train_y_var)
        model.fit(n_iter=200)
        
        train_ypred = model.predict(train_x).detach().cpu().numpy()
        test_ypred = model.predict(test_x).detach().cpu().numpy()
        
        train_rho = pearsonr(train_ypred, train_y)[0]
        test_rho = pearsonr(test_ypred, test_y)[0]
        
        w = kernel.w.detach().cpu().numpy()
        print(w, train_rho, test_rho)
        assert(w[0] < w[1])
        assert(w[-1] < w[-2])
        assert(train_rho > 0.9)
        assert(test_rho > 0.6)
    
    def test_epik_smn1_prediction(self):
        train_x, train_y, test_x, test_y, train_y_var = get_smn1_data(n=2000)
        
        starting_log_lambdas = get_tensor([-1.01, -2.02, -3.04, -4.07, -5.11, -6.16, -7.22])
        kernel = SkewedVCKernel(n_alleles=4, seq_length=7, tau=1,
                                train_p=False, train_lambdas=False,
                                starting_log_lambdas=starting_log_lambdas)
        model = EpiK(kernel, likelihood_type='Gaussian')
        model.fit(train_x, train_y, y_var=train_y_var, n_iter=0)
        
        train_ypred = model.predict(train_x).detach().numpy()
        test_ypred = model.predict(test_x).detach().numpy()
        
        train_rho = pearsonr(train_ypred, train_y)[0]
        test_rho = pearsonr(test_ypred, test_y)[0]
        
        assert(train_rho > 0.9)
        assert(test_rho > 0.6)
    
    def test_epik_smn1_gpu_partition(self):
        train_x, train_y, test_x, test_y, train_y_var = get_smn1_data(n=2000)
        l, a = 7, 4
        n_devices, output_device = 1, torch.device('cuda:0')
        
        lambdas_prior = LambdasFlatPrior(seq_length=l)
        p_prior = AllelesProbPrior(seq_length=l, n_alleles=a)
        kernel = SkewedVCKernel(n_alleles=a, seq_length=l, q=0.7,
                                lambdas_prior=lambdas_prior, p_prior=p_prior)
        model = EpiK(kernel, output_device=output_device, n_devices=n_devices,
                     learning_rate=0.02)
        model.set_data(train_x, train_y, y_var=train_y_var)
        
        model.partition_size = 1000
        # model.optimize_partition_size()
        # print(model.partition_size)
        model.fit(n_iter=50)
        
        train_ypred = model.predict(train_x).cpu().detach().numpy()
        test_ypred = model.predict(test_x).cpu().detach().numpy()
        
        train_rho = pearsonr(train_ypred, train_y)[0]
        test_rho = pearsonr(test_ypred, test_y)[0]
        print(train_rho, test_rho)
        assert(train_rho > 0.9)
        assert(test_rho > 0.6)
            
    def test_recover_site_weights(self):
        ws = np.array([0.5, 0.9, 0.05, 0.9, 0.1])
        l, a = ws.shape[0], 5
        ps = np.array([[1-w] + [w/(a-1)]*(a-1) for w in ws])
        logit = np.log((1-ps) / ps)
        lambdas = np.exp(np.append([-10], -5*np.arange(l)))
        vc = VCregression()
        vc.init(l, a, ps=ps)
        
        # Data
        data = vc.simulate(lambdas=lambdas, sigma=0, p_missing=0)
        data = data.loc[['0' not in x for x in data.index.values], :]
        x = seq_to_one_hot(data.index.values, alleles=['0', '1', '2', '3', '4']) 
        y = data['y'].values
        y = (y - y.mean())/ y.std()
        sigma = 0.1
        y_obs = np.random.normal(y, sigma)
        y_var = sigma**2 * np.ones(y.shape[0])
        
        # Model fit
        kernel = SiteProductKernel(n_alleles=a, seq_length=l)
        model = EpiK(kernel, likelihood_type='Gaussian')
        model.fit(x, y_obs, y_var=y_var, n_iter=200, learning_rate=0.05)
        
        ypred = model.predict(x).detach().numpy()
        w_hat = kernel.w.detach().numpy()
        
        rho1 = pearsonr(ypred, y)[0]
        rho2 = pearsonr(ypred, y_obs)[0]
        rho3 = pearsonr(logit[:, 1], w_hat)[0]
        print(w_hat, logit[:, 1])
        assert(w_hat[0] > w_hat[1])
        assert(w_hat[-1] > w_hat[-2])
        assert(rho1 > 0.9)
        assert(rho2 > 0.9)
        assert(rho3 > 0.9)

    def test_recover_p(self):
        np.random.seed(0)
        vc = VCregression()
        l = 5
        ps = np.vstack([[0.25, 0.025, 0.025, 0.5]] * 4).T
        l = ps.shape[0]
        vc.init(l, 4, ps=ps)
        lambdas = np.append([0], 1e7 * 10. ** (-np.arange(l)))
        data = vc.simulate(lambdas=lambdas, sigma=0, p_missing=0)
        seqs, y, y_var = data.index.values, data['y'].values, data['var'].values
        sigma = 0.01 * y.std()
        y = np.random.normal(y, sigma)
        print(y, sigma)
        y_var = sigma * np.ones(y.shape[0])
        
        X = seq_to_one_hot(seqs, alleles=['0', '1', '2', '3'])
        train_x, train_y, test_x, test_y, train_y_var = split_training_test(X, y, y_var, dtype=torch.float64)
        
        # Train skewed
        kernel1 = SkewedVCKernel(n_alleles=4, seq_length=l, tau=0.2, train_p=True,
                                # starting_p=get_tensor(ps, dtype=torch.float64),
                                dtype=torch.float64
                                )
        model = EpiK(kernel1, likelihood_type='Gaussian', dtype=torch.float64)
        model.fit(train_x, train_y, y_var=train_y_var, n_iter=150, learning_rate=0.01)
        test_ypred = model.predict(test_x).detach().numpy()
        test_rho1 = pearsonr(test_ypred, test_y)[0]
        
        # Train
        kernel2 = SkewedVCKernel(n_alleles=4, seq_length=l, tau=0.2, train_p=False,
                                 dtype=torch.float64)
        model = EpiK(kernel2, likelihood_type='Gaussian', dtype=torch.float64)
        model.fit(train_x, train_y, y_var=train_y_var, n_iter=150, learning_rate=0.02)
        test_ypred = model.predict(test_x).detach().numpy()
        test_rho2 = pearsonr(test_ypred, test_y)[0]
        

        print(test_rho1, test_rho2)        
        print(kernel1.p)
        print(kernel1.lambdas.detach().numpy())
        print(kernel2.lambdas.detach().numpy())
        # print(ps_to_variances(ps))
        # print(ps_to_variances(kernel.p.detach().numpy))
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'ModelsTests']
    unittest.main()
