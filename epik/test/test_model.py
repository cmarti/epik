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
        model = EpiK(kernel, likelihood_type='Gaussian',
                     alleles=['A', 'B'])
        
        X = np.array(['AA', 'AB', 'BA', 'BB'])
        y = torch.tensor([0.2, 1.1, 0.5, 1.5])
        model.fit(X, y, n_iter=100)
        ypred = model.predict(X)
        assert(pearsonr(ypred, y)[0] > 0.9)
        
    def test_epik_basic_loo(self):
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2, train_p=False)
        model = EpiK(kernel, likelihood_type='Gaussian',
                     alleles=['A', 'B'])
        
        train_X, test_X = np.array(['AA', 'AB', 'BA']), np.array(['BB'])
        train_y, test_y = torch.tensor([0.2, 1.1, 0.5]), torch.tensor([1.5])
        model.fit(train_X, train_y, n_iter=1000)
        
        print('lambdas')
        # print(model.kernel.log_lda_alpha.detach(), model.kernel.log_lda_beta.detach())
        print(model.kernel.log_lda.detach().numpy())
        
        test_predy = model.predict(test_X)
        print(test_y, test_predy)
        
        train_predy = model.predict(train_X)
        print(train_predy, train_y)
        assert(pearsonr(train_predy, train_y)[0] > 0.9)
    
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
    
    def test_epik_smn1(self):
        n = 2000
        data = pd.read_csv(join(TEST_DATA_DIR, 'smn1data.csv'),
                           header=None, index_col=0, names=['m', 'std'])
        data = data.loc[[x[3] == 'U' for x in data.index], :]
        data.index = [x[:3] + x[4:] for x in data.index]
        seqs = data.index.values
        np.random.shuffle(seqs)
        data = data.loc[seqs, :]
        
        train, test = data.iloc[:n, :], data.iloc[-2000:, :]
        train_X, train_y = train.index.values, train['m'].values
        test_X, test_y = test.index.values, test['m'].values
        train_y_var = (data['std'] ** 2).values
        
        kernel = SkewedVCKernel(n_alleles=4, seq_length=7, train_p=True,
                                force_exp_decay=False)
        model = EpiK(kernel, likelihood_type='Gaussian', train_mean=True,
                     alleles=['A', 'C', 'G', 'U'])
        model.fit(train_X, train_y, y_var=train_y_var, n_iter=100)
        
        
        train_ypred = model.predict(train_X)
        print(train_ypred.min(), train_ypred.max())
        print(pearsonr(train_ypred, train_y)[0])
        
        test_ypred = model.predict(test_X)
        print(test_ypred.min(), test_ypred.max())
        print(test_y.min(), test_y.max())
        print(pearsonr(test_ypred, test_y)[0])
        
        print(model.kernel.p)
        print(model.kernel.log_lda)
        print(model.kernel.lambdas)
        
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'ModelsTests.test_epik_smn1']
    unittest.main()
