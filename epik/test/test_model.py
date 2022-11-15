#!/usr/bin/env python
import unittest

import torch
import pandas as pd
import numpy as np

from os.path import join

from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel

from epik.src.settings import TEST_DATA_DIR
from epik.src.kernel import SkewedVCKernel
from epik.src.model import EpiK
from scipy.stats.stats import pearsonr
from epik.src.utils import seq_to_one_hot


def get_smn1_data(n, seed=0):
    np.random.seed(seed)
    data = pd.read_csv(join(TEST_DATA_DIR, 'smn1data.csv'),
                           header=None, index_col=0, names=['m', 'std'])
    data = data.loc[[x[3] == 'U' for x in data.index], :]
    data.index = [x[:3] + x[4:] for x in data.index]
    
    alleles = ['A', 'C', 'G', 'U']
    X, y = seq_to_one_hot(data.index.values, alleles=alleles), data['m'].values
    ymean, ystd = y.mean(), y.std()
    y = (y -ymean) / ystd
    y_var = (data['std'].values / ystd) ** 2
    
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
    return(train_x, train_y, test_x, test_y, train_y_var)


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
    
    def test_epik_smn1(self):
        train_x, train_y, test_x, test_y, train_y_var = get_smn1_data(n=2000)
        
        kernel = SkewedVCKernel(n_alleles=4, seq_length=7, train_p=True, tau=1)
        model = EpiK(kernel, likelihood_type='Gaussian')
        model.fit(train_x, train_y, y_var=train_y_var,
                  n_iter=100, learning_rate=0.02)
        
        train_ypred = model.predict(train_x).detach().numpy()
        test_ypred = model.predict(test_x).detach().numpy()
        
        train_rho = pearsonr(train_ypred, train_y)[0]
        test_rho = pearsonr(test_ypred, test_y)[0]
        
        assert(train_rho > 0.9)
        assert(test_rho > 0.6)
    
    def test_epik_smn1_gpu(self):
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
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'ModelsTests.test_epik_smn1']
    unittest.main()
