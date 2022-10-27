#!/usr/bin/env python
import unittest

import pandas as pd
import numpy as np
import torch

from os.path import join

from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel

from epik.src.settings import TEST_DATA_DIR
from epik.src.kernel import SkewedVCKernel
from epik.src.model import EpiK
from scipy.stats.stats import pearsonr
from epik.src.utils import seq_to_one_hot


def get_smn1_data(n):
    data = pd.read_csv(join(TEST_DATA_DIR, 'smn1data.csv'),
                           header=None, index_col=0, names=['m', 'std'])
    data = data.loc[[x[3] == 'U' for x in data.index], :]
    data.index = [x[:3] + x[4:] for x in data.index]
    
    n = 1000
    p = n / data.shape[0]
    ps = np.random.uniform(size=data.shape[0])
    test = data.loc[ps < p, :]
    train = data.loc[ps > (1 - p), :]
    
    alleles = ['A', 'C', 'G', 'U']
    train_x, train_y = seq_to_one_hot(train.index.values, alleles=alleles), train['m'].values
    test_x, test_y = seq_to_one_hot(test.index.values, alleles=alleles), test['m'].values
    train_y_var = (train['std'] ** 2).values
    return(train_x, train_y, test_x, test_y, train_y_var)


class ModelsTests(unittest.TestCase):
    def test_one_hot_encoding(self):
        X = np.array(['AA', 'AB', 'BA', 'BB'])
        x = seq_to_one_hot(X).numpy()
        onehot = np.array([[1, 0, 1, 0],
                           [1, 0, 0, 1],
                           [0, 1, 1, 0],
                           [0, 1, 0, 1.]])
        assert(np.allclose(x - onehot, 0))
        
    def test_epik_basic(self):
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2)
        model = EpiK(kernel, likelihood_type='Gaussian',
                     alleles=['A', 'B'])
        
        X = seq_to_one_hot(np.array(['AA', 'AB', 'BA', 'BB']))
        y = torch.tensor([0.2, 1.1, 0.5, 1.5])
        model.fit(X, y, n_iter=100)
        ypred = model.predict(X)
        assert(pearsonr(ypred, y)[0] > 0.9)
        
    def test_epik_basic_loo(self):
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2, train_p=False)
        model = EpiK(kernel, likelihood_type='Gaussian', train_mean=True,
                     alleles=['A', 'B'])
        
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
        
        X = np.array(['AA', 'AB', 'BA', 'BB'])
        y = torch.tensor([0.2, 1.1, 0.5, 1.5])
        model.fit(X, y, n_iter=100)
        ypred = model.predict(X)
        assert(ypred.shape[0] == 4)
        assert(pearsonr(ypred, y)[0] > 0.9)
        
    def test_epik_basic_RBF(self):
        kernel = ScaleKernel(RBFKernel())
        model = EpiK(kernel, likelihood_type='Gaussian', train_mean=True)
        
        X = np.array(['AA', 'AB', 'BA', 'BB'])
        y = torch.tensor([0.2, 1.1, 0.5, 1.5])
        model.fit(X, y, n_iter=200)
        ypred = model.predict(X)
        print(ypred)
    
    def test_rbf_smn1(self):
        train_x, train_y, test_x, test_y, train_y_var = get_smn1_data(n=1000)
        
        kernel = ScaleKernel(RBFKernel())
        model = EpiK(kernel, likelihood_type='Gaussian', train_mean=True,
                     alleles=['A', 'C', 'G', 'U'])
        model.fit(train_x, train_y, y_var=train_y_var,
                  n_iter=100, learning_rate=0.05)
        
        train_ypred = model.predict(train_x)
        test_ypred = model.predict(test_x).detach().numpy()
        
        train_rho = pearsonr(train_ypred, train_y)[0]
        test_rho = pearsonr(test_ypred, test_y)[0]
        print(train_rho, test_rho)
        assert(train_rho > 0.7)
        assert(test_rho > 0.6)
        
    
    def test_epik_smn1(self):
        train_x, train_y, test_x, test_y, train_y_var = get_smn1_data(n=4000)
        
        kernel = SkewedVCKernel(n_alleles=4, seq_length=7, train_p=True,
                                force_exp_decay=False)
        model = EpiK(kernel, likelihood_type='Gaussian', train_mean=True,
                     alleles=['A', 'C', 'G', 'U'])
        model.fit(train_x, train_y, y_var=train_y_var,
                  n_iter=100, learning_rate=0.02)
        
        train_ypred = model.predict(train_x)
        test_ypred = model.predict(test_x).detach().numpy()
        
        train_rho = pearsonr(train_ypred, train_y)[0]
        test_rho = pearsonr(test_ypred, test_y)[0]
        print(train_rho, test_rho)
        assert(train_rho > 0.9)
        assert(test_rho > 0.6)
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'ModelsTests']
    unittest.main()
