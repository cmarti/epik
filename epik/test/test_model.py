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

from epik.src.settings import TEST_DATA_DIR, BIN_DIR
from epik.src.utils import (seq_to_one_hot, get_tensor, split_training_test,
                            get_full_space_one_hot, one_hot_to_seq)
from epik.src.model import EpiK
from epik.src.kernel import SkewedVCKernel, HetRBFKernel
from epik.src.priors import (LambdasExpDecayPrior, AllelesProbPrior,
                             LambdasDeltaPrior, LambdasFlatPrior, RhosPrior)
from epik.src.keops import RhoKernel, VarianceComponentKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.kernels.rbf_kernel import RBFKernel


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
                                     log_lambdas0=log_lambdas0)
    model = EpiK(kernel)
    return(alpha, l, log_lambdas0, model.simulate_dataset(X, sigma=sigma, ptrain=ptrain))


def get_rho_random_landscape_data(sigma=0, ptrain=0.8):
    rho0 = torch.tensor([[0.1, 0.5, 0.05, 0.25, 0.7]]).T
    logit_rho0 = torch.log(rho0 / (1 - rho0))
    alpha, l = 4, logit_rho0.shape[0]
    X = get_full_space_one_hot(seq_length=l, n_alleles=alpha)
    
    kernel = RhoKernel(n_alleles=alpha, seq_length=l, logit_rho0=logit_rho0)
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
        model = EpiK(VarianceComponentKernel(n_alleles=alpha, seq_length=l),
                     track_progress=True)
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=100)
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
        model = EpiK(VarianceComponentKernel(n_alleles=alpha, seq_length=l,
                                             log_lambdas0=log_lambdas0))
        model.set_data(train_x, train_y, train_y_var)
        ypred = model.predict(test_x).detach()
        r2 = pearsonr(ypred, test_y)[0] ** 2
        assert(r2 > 0.9)
    
    def test_epik_keops(self):
        # Simulate from prior distribution
        alpha, l, logit_rho0, data = get_rho_random_landscape_data(sigma=0, ptrain=0.9)
        train_x, train_y, test_x, test_y, train_y_var = data
        
        # Train new model
        kernel = RhoKernel(alpha, l)
        model = EpiK(kernel, track_progress=True)
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=50)
        logit_rho = kernel.logit_rho.detach().numpy().flatten() 
        r = pearsonr(logit_rho, logit_rho0.flatten())[0]
        assert(r > 0.8)
        
        # Predict unobserved sequences
        kernel = RhoKernel(alpha, l, logit_rho0=logit_rho0)
        model = EpiK(kernel)
        model.set_data(train_x, train_y, train_y_var)
        ypred = model.predict(test_x).detach()
        r2 = pearsonr(ypred, test_y)[0] ** 2
        assert(r2 > 0.9)
    
    def test_epik_keops_gpu(self):
        # Simulate from prior distribution
        alpha, l, logit_rho0, data = get_rho_random_landscape_data(sigma=0, ptrain=0.9)
        train_x, train_y, test_x, test_y, train_y_var = data
        
        # Train new model
        kernel = RhoKernel(alpha, l)
        model = EpiK(kernel, track_progress=True, device=torch.device('cuda:0'))
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=50)
        logit_rho = kernel.logit_rho.cpu().detach().numpy().flatten() 
        r = pearsonr(logit_rho, logit_rho0.flatten())[0]
        assert(r > 0.8)
        
        # # Predict unobserved sequences
        kernel = RhoKernel(alpha, l, logit_rho0=logit_rho0)
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
    
    def test_epik_het(self):
        train_x, train_y, test_x, test_y, train_y_var = get_smn1_data(n=1000)
        l, alpha = 7, 4
        track_progress = False
        
        kernel = ScaleKernel(RBFKernel(ard_num_dims=1))
        model = EpiK(kernel, track_progress=track_progress, learning_rate=0.2)
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=200)
        ypred = model.predict(test_x).detach()
        r2_1 = pearsonr(ypred, test_y)[0] ** 2
        print(r2_1)
        
        kernel = HetRBFKernel(alpha, l, dims=1, train_het=False)
        model = EpiK(kernel, track_progress=track_progress)
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=100)
        ypred = model.predict(test_x).detach()
        r2_2 = pearsonr(ypred, test_y)[0] ** 2
        print(r2_2)
        print(kernel.theta)
        print(kernel.theta0)
        
        kernel = HetRBFKernel(alpha, l, dims=1)
        model = EpiK(kernel, track_progress=track_progress)
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=100)
        ypred = model.predict(test_x).detach()
        r2_2 = pearsonr(ypred, test_y)[0] ** 2
        print(r2_2)
        print(kernel.theta)
        print(kernel.theta0)
        # # exit()
        kernel = ScaleKernel(RBFKernel(ard_num_dims=alpha * l))
        model = EpiK(kernel, track_progress=track_progress, learning_rate=0.2)
        model.set_data(train_x, train_y, train_y_var)
        model.fit(n_iter=200)
        ypred = model.predict(test_x).detach()
        r2_3 = pearsonr(ypred, test_y)[0] ** 2
        print(r2_3)
        
        kernel = HetRBFKernel(alpha, l, dims=alpha*l)
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
    import sys;sys.argv = ['', 'ModelsTests.test_epik_bin']
    unittest.main()
