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


class ModelsTests(unittest.TestCase):
    def test_one_hot_encoding(self):
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2)
        model = EpiK(kernel, likelihood_type='Gaussian')
        X = np.array(['AA', 'AB', 'BA', 'BB'])
        x = model.seq_to_one_hot(X).numpy()
        onehot = np.array([[1, 0, 1, 0],
                           [1, 0, 0, 1],
                           [0, 1, 1, 0],
                           [0, 1, 0, 1.]])
        assert(np.allclose(x - onehot, 0))
        
    def test_epik_basic(self):
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2)
        model = EpiK(kernel, likelihood_type='Gaussian')
        
        X = np.array(['AA', 'AB', 'BA', 'BB'])
        y = torch.tensor([0.2, 1.1, 0.5, 1.5])
        model.fit(X, y, n_iter=100)
        ypred = model.predict(X)
        assert(ypred.shape[0] == 4)
    
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
        
    def test_epik_basic_RBF(self):
        kernel = ScaleKernel(RBFKernel())
        model = EpiK(kernel, likelihood_type='Gaussian', train_mean=True)
        
        X = np.array(['AA', 'AB', 'BA', 'BB'])
        y = torch.tensor([0.2, 1.1, 0.5, 1.5])
        model.fit(X, y, n_iter=200)
        ypred = model.predict(X)
        print(ypred)
    
    def test_epik_smn1_RBFkernel(self):
        data = pd.read_csv(join(TEST_DATA_DIR, 'smn1data.csv'),
                           header=None, index_col=0, names=['m', 'std'])
        data = data.loc[[x[3] == 'U' for x in data.index], :]
        data.index = [x[:3] + x[4:] for x in data.index]
        data = data.loc[np.random.choice(data.index, size=1000), :]
        
        X = data.index.values
        y = data['m'].values
        y_var = (data['std'] ** 2).values
        
        # kernel = SkewedVCKernel(n_alleles=4, seq_length=7)
        output_device = None
        # output_device = torch.device('cuda:0')
        
        kernel = ScaleKernel(RBFKernel())
        model = EpiK(kernel, likelihood_type='Gaussian', train_mean=True,
                     output_device=output_device)
        model.fit(X, y, y_var=y_var, n_iter=1000)
        ypred = model.predict(X)
        print(ypred)
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'ModelsTests']
    unittest.main()
