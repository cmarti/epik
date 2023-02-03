#!/usr/bin/env python
import unittest
import sys

import torch
import pandas as pd
import numpy as np

from os.path import join
from subprocess import check_call
from tempfile import NamedTemporaryFile

from scipy.stats.stats import pearsonr
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel

from epik.src.settings import TEST_DATA_DIR, BIN_DIR
from epik.src.kernel import SkewedVCKernel, VCKernel, ExponentialKernel
from epik.src.model import EpiK
from epik.src.utils import seq_to_one_hot, get_tensor, split_training_test,\
    ps_to_variances
from gpmap.src.inference import VCregression


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
        print('a')
        output = [get_tensor(a, dtype=dtype) for a in output]
    return(output)


class ModelsTests(unittest.TestCase):
    def test_epik_basic(self):
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2)
        model = EpiK(kernel, likelihood_type='Gaussian')
        
        X = seq_to_one_hot(np.array(['AA', 'AB', 'BA', 'BB']))
        y = torch.tensor([0.2, 1.1, 0.5, 1.5])
        model.fit(X, y, n_iter=100)
        ypred = model.predict(X)
        assert(pearsonr(ypred, y)[0] > 0.9)
        
    def test_epik_basic_loo(self):
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2, train_p=False)
        model = EpiK(kernel, likelihood_type='Gaussian')
        
        train_X = seq_to_one_hot(np.array(['AA', 'AB', 'BA']), alleles=['A', 'B'])
        test_X = seq_to_one_hot(np.array(['BB']), alleles=['A', 'B'])
        train_y = np.array([0.2, 1.1, 0.5])
        test_y = [1.5]
        model.fit(train_X, train_y, n_iter=100)
        
        test_predy = model.predict(test_X)
        print(test_y, test_predy)
        
        train_predy = model.predict(train_X)
        train_rho = pearsonr(train_predy, train_y)[0]
        print(train_predy, train_y, train_rho)
        assert(train_rho > 0.9)
    
    def test_epik_basic_gpu(self):
        output_device = torch.device('cuda:0') 
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2)
        model = EpiK(kernel, likelihood_type='Gaussian',
                     output_device=output_device)
        
        X = seq_to_one_hot(np.array(['AA', 'AB', 'BA', 'BB']))
        y = torch.tensor([0.2, 1.1, 0.5, 1.5])
        model.fit(X, y, n_iter=100)
        ypred = model.predict(X)
        assert(ypred.shape[0] == 4)
        assert(pearsonr(ypred.cpu().numpy(), y.cpu().numpy())[0] > 0.9)
        
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
        train_x, train_y, test_x, test_y, train_y_var = get_smn1_data(n=2000,
                                                                      dtype=torch.float64)
        kernel = SkewedVCKernel(n_alleles=4, seq_length=7, train_p=True,
                                q=0.5, tau=1, lambdas_prior='monotonic_decay')
        model = EpiK(kernel, likelihood_type='Gaussian', dtype=torch.float64)
        model.fit(train_x, train_y, y_var=train_y_var,
                  n_iter=100, learning_rate=0.02)
        
        train_ypred = model.predict(train_x).detach().numpy()
        test_ypred = model.predict(test_x).detach().numpy()
        
        train_rho = pearsonr(train_ypred, train_y)[0]
        test_rho = pearsonr(test_ypred, test_y)[0]
        
        assert(train_rho > 0.9)
        assert(test_rho > 0.6)
        
    def test_epik_vc_smn1(self):
        train_x, train_y, test_x, test_y, train_y_var = get_smn1_data(n=1000)
        
        kernel = VCKernel(n_alleles=4, seq_length=7, tau=0.2)
        model = EpiK(kernel, likelihood_type='Gaussian')
        model.fit(train_x, train_y, y_var=train_y_var,
                  n_iter=100, learning_rate=0.02)
        
        train_ypred = model.predict(train_x).detach().numpy()
        test_ypred = model.predict(test_x).detach().numpy()
        
        train_rho = pearsonr(train_ypred, train_y)[0]
        test_rho = pearsonr(test_ypred, test_y)[0]
        
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
    
    def test_epik_smn1_skewed_vc_gpu(self):
        train_x, train_y, test_x, test_y, train_y_var = get_smn1_data(n=2000)
        output_device = torch.device('cuda:0')
        
        kernel = SkewedVCKernel(n_alleles=4, seq_length=7, train_p=True, tau=0.2)
        model = EpiK(kernel, likelihood_type='Gaussian',
                     output_device=output_device)
        model.fit(train_x, train_y, y_var=train_y_var,
                  n_iter=100, learning_rate=0.01)
        
        train_ypred = model.predict(train_x).cpu().detach().numpy()
        test_ypred = model.predict(test_x).cpu().detach().numpy()
        
        train_rho = pearsonr(train_ypred, train_y)[0]
        test_rho = pearsonr(test_ypred, test_y)[0]
        assert(train_rho > 0.9)
        assert(test_rho > 0.6)
    
    def test_epik_smn1_vc_gpu(self):
        train_x, train_y, test_x, test_y, train_y_var = get_smn1_data(n=2000)
        output_device = torch.device('cuda:0')
        
        kernel = VCKernel(n_alleles=4, seq_length=7, tau=0.2)
        model = EpiK(kernel, likelihood_type='Gaussian',
                     output_device=output_device)
        model.fit(train_x, train_y, y_var=train_y_var,
                  n_iter=100, learning_rate=0.01)
        
        train_ypred = model.predict(train_x).cpu().detach().numpy()
        test_ypred = model.predict(test_x).cpu().detach().numpy()
        
        train_rho = pearsonr(train_ypred, train_y)[0]
        test_rho = pearsonr(test_ypred, test_y)[0]
        assert(train_rho > 0.9)
        assert(test_rho > 0.6)
    
    def test_epik_smn1_exponential_gpu(self):
        train_x, train_y, test_x, test_y, train_y_var = get_smn1_data(n=2000)
        output_device = torch.device('cuda:0')
        
        kernel = ExponentialKernel(n_alleles=4, seq_length=7, train_p=True)
        model = EpiK(ScaleKernel(kernel), likelihood_type='Gaussian',
                     output_device=output_device)
        model.fit(train_x, train_y, y_var=train_y_var,
                  n_iter=100, learning_rate=0.05)
        
        train_ypred = model.predict(train_x).cpu().detach().numpy()
        test_ypred = model.predict(test_x).cpu().detach().numpy()
        
        train_rho = pearsonr(train_ypred, train_y)[0]
        test_rho = pearsonr(test_ypred, test_y)[0]

        assert(test_ypred.shape[0] == test_x.shape[0])
        assert(train_rho > 0.9)
        assert(test_rho > 0.75)
    
    def test_epik_smn1_gpu_partition(self):
        partition_size = 100

        train_x, train_y, test_x, test_y, train_y_var = get_smn1_data(n=2000)
        n_devices, output_device = 1, torch.device('cuda:0')
        
        kernel = SkewedVCKernel(n_alleles=4, seq_length=7, train_p=True, tau=0.2)
        model = EpiK(kernel, likelihood_type='Gaussian',
                     output_device=output_device, n_devices=n_devices)
        
        # model.optimize_partition_size(train_x, train_y, y_var=train_y_var)
        model.fit(train_x, train_y, y_var=train_y_var,
                  n_iter=50, learning_rate=0.02, partition_size=partition_size)
        
        train_ypred = model.predict(train_x, partition_size=partition_size).cpu().detach().numpy()
        test_ypred = model.predict(test_x, partition_size=partition_size).cpu().detach().numpy()
        
        train_rho = pearsonr(train_ypred, train_y)[0]
        test_rho = pearsonr(test_ypred, test_y)[0]
        assert(train_rho > 0.9)
        assert(test_rho > 0.6)
    
    def test_epik_bin(self):
        bin_fpath = join(BIN_DIR, 'EpiK.py')
        data_fpath = join(TEST_DATA_DIR, 'smn1.train.csv')
        xpred_fpath = join(TEST_DATA_DIR, 'smn1.test.txt')
        
        with NamedTemporaryFile() as fhand:
            out_fpath = fhand.name
        
            # Check help
            cmd = [sys.executable, bin_fpath, '-h']
            check_call(cmd)
            
            # Model fitting
            cmd = [sys.executable, bin_fpath, data_fpath, '-o', out_fpath, '-n', '50']
            check_call(cmd)
            
            # Predict test sequences
            cmd.extend(['-p', xpred_fpath])
            check_call(cmd)
            
            # Predict test sequences with variable ps
            cmd.extend(['-k', 'sVC', '--train_p', '--use_float64'])
            check_call(cmd)
            
            # Predict test sequences with variable ps using GPU
            cmd.extend(['--gpu'])
            check_call(cmd)
            
    
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
    import sys;sys.argv = ['', 'ModelsTests.test_recover_p']
    unittest.main()

