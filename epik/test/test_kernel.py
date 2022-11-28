#!/usr/bin/env python
import unittest

import numpy as np
import matplotlib.pyplot as plt
import torch

from epik.src.kernel import SkewedVCKernel,  VCKernel
from epik.src.utils import seq_to_one_hot, get_tensor
from epik.src.settings import TEST_DATA_DIR
from os.path import join


class KernelsTests(unittest.TestCase):
    def test_calc_polynomial_coeffs(self):
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2)
        lambdas = kernel.calc_eigenvalues()
        V = torch.stack([torch.pow(lambdas, i) for i in range(3)], 1)

        B = kernel.coeffs
        P = np.array(torch.matmul(B, V))
        assert(np.allclose(P, np.eye(3), atol=1e-4))
    
    def test_vc_kernel(self):
        kernel = VCKernel(n_alleles=2, seq_length=2)
        x = torch.tensor([[1, 0, 1, 0],
                          [0, 1, 1, 0],
                          [1, 0, 0, 1],
                          [0, 1, 0, 1]], dtype=torch.float32)

        # k=0        
        lambdas = torch.tensor([1, 0, 0], dtype=torch.float32)
        cov = kernel._forward(x, x, lambdas)
        assert(np.allclose(cov, 1))
        
        # k=1        
        lambdas = torch.tensor([0, 1, 0], dtype=torch.float32)
        cov = kernel._forward(x, x, lambdas).numpy()
        k1 = np.array([[2, 0, 0, -2],
                       [0, 2, -2, 0],
                       [0, -2, 2, 0],
                       [-2, 0, 0, 2]], dtype=np.float32)
        assert(np.abs(cov - k1).mean() < 1e-4)
        
        # k=2
        lambdas = torch.tensor([0, 0, 1], dtype=torch.float32)
        cov = kernel._forward(x, x, lambdas).numpy()
        k2 = np.array([[1, -1, -1, 1],
                       [-1, 1, 1, -1],
                       [-1, 1, 1, -1],
                       [1, -1, -1, 1]], dtype=np.float32)
        assert(np.abs(cov - k2).mean() < 1e-4)
    
    def test_skewed_vc_kernel(self):
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2)
        log_p = torch.log(torch.tensor([[0.5, 0.5, 0.00001],
                                        [0.5, 0.5, 0.00001]], dtype=torch.float32))
        x = torch.tensor([[1, 0, 1, 0],
                          [0, 1, 1, 0],
                          [1, 0, 0, 1],
                          [0, 1, 0, 1]], dtype=torch.float32)

        # k=0        
        lambdas = torch.tensor([1, 0, 0], dtype=torch.float32)
        cov = kernel._forward(x, x, lambdas, log_p)
        assert(np.allclose(cov, 1))
        
        # k=1        
        lambdas = torch.tensor([0, 1, 0], dtype=torch.float32)
        cov = kernel._forward(x, x, lambdas, log_p).numpy()
        k1 = np.array([[2, 0, 0, -2],
                       [0, 2, -2, 0],
                       [0, -2, 2, 0],
                       [-2, 0, 0, 2]], dtype=np.float32)
        assert(np.abs(cov - k1).mean() < 1e-4)
        
        # k=2
        lambdas = torch.tensor([0, 0, 1], dtype=torch.float32)
        cov = kernel._forward(x, x, lambdas, log_p).numpy()
        k2 = np.array([[1, -1, -1, 1],
                       [-1, 1, 1, -1],
                       [-1, 1, 1, -1],
                       [1, -1, -1, 1]], dtype=np.float32)
        assert(np.abs(cov - k2).mean() < 1e-4)
    
        # Longer seqs
        kernel = SkewedVCKernel(n_alleles=2, seq_length=5)
        log_p = torch.log(torch.tensor([[0.5, 0.5, 0.00001],
                                        [0.5, 0.5, 0.00001],
                                        [0.5, 0.5, 0.00001],
                                        [0.5, 0.5, 0.00001],
                                        [0.5, 0.5, 0.00001],
                                        [0.5, 0.5, 0.00001]], dtype=torch.float32))
        x = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                          [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                          [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
                          [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
                          [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
                          [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
                          [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]], dtype=torch.float32)

        lambdas = torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.float32)
        cov = kernel._forward(x[:1], x[1:], lambdas, log_p).numpy()
        assert(np.allclose(cov[0][:-1] - cov[0][1:], 2, atol=0.01))
        
        # More than 2 alleles
        kernel = SkewedVCKernel(n_alleles=3, seq_length=2)
        log_p = torch.log(torch.tensor([[0.33, 0.33, 0.33, 0.00001],
                                        [0.33, 0.33, 0.33, 0.00001]], dtype=torch.float32))
        x = torch.tensor([[1, 0, 0, 1, 0, 0],
                          [0, 1, 0, 1, 0, 0],
                          [0, 0, 1, 1, 0, 0],
                          [1, 0, 0, 0, 1, 0],
                          [0, 1, 0, 0, 1, 0],
                          [0, 0, 1, 0, 1, 0],
                          [1, 0, 0, 0, 0, 1],
                          [0, 1, 0, 0, 0, 1],
                          [0, 0, 1, 0, 0, 1]], dtype=torch.float32)

        lambdas = torch.tensor([0, 1, 0], dtype=torch.float32)
        cov = kernel._forward(x[:1], x, lambdas, log_p).numpy()
        k1 = np.array([4, 1, 1, 1, -2, -2, 1, -2, -2])
        assert(np.abs(cov[0] - k1).mean() < 1e-4)
        
    def test_skewed_vc_kernel_diag(self):
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2)
        log_p = torch.log(torch.tensor([[0.45, 0.45, 0.1],
                                        [0.45, 0.45, 0.1]], dtype=torch.float32))
        x = torch.tensor([[1, 0, 1, 0],
                          [0, 1, 1, 0],
                          [1, 0, 0, 1],
                          [0, 1, 0, 1]], dtype=torch.float32)

        lambdas = torch.tensor([1, 0.5, 0.1], dtype=torch.float32)
        cov = kernel._forward(x, x, lambdas, log_p, diag=False)
        var = kernel._forward(x, x, lambdas, log_p, diag=True)
        assert(np.allclose(cov.numpy().diagonal(), var.numpy()))
        
    def test_kernel_params(self):
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2,
                                train_p=False, train_lambdas=False,
                                starting_log_lambdas=[0, -10])
        x = torch.tensor([[1, 0, 1, 0],
                          [0, 1, 1, 0],
                          [1, 0, 0, 1],
                          [0, 1, 0, 1]], dtype=torch.float32)
        cov = kernel.forward(x, x)
        assert(np.allclose(cov[0], [2, 0, 0, -2], atol=0.01))
        
    def test_vc_kernels_variable_lengths(self):
        for alleles in [['A', 'B'], ['A', 'B', 'C']]:
            alpha = len(alleles)
            for l in range(2, 8):
                seqs = np.array(['A' * i + 'B' * (l-i) for i in range(l)])
                train_x = seq_to_one_hot(seqs, alleles=alleles)
                for i in range(l):
                    starting_log_lambdas = -10 * np.ones(l)
                    starting_log_lambdas[i] = 0
                    starting_log_lambdas = get_tensor(starting_log_lambdas)
                 
                    ker = SkewedVCKernel(alpha, l, q=0.7, tau=.1,
                                         starting_log_lambdas=starting_log_lambdas)
                    cov1 = ker.forward(train_x, train_x).detach().numpy()
                    
                    ker = VCKernel(alpha, l, tau=.1,
                                   starting_log_lambdas=starting_log_lambdas)
                    cov2 = ker.forward(train_x, train_x).detach().numpy()
                    
                    mae = np.abs(cov1 - cov2).mean()
                    assert(np.allclose(mae, 0, atol=1))
                    
    def test_plot_vc_cov_d_function(self):
        fig, axes = plt.subplots(1, 1, figsize=(5, 4))
        
        alleles = ['A', 'B']
        alpha = len(alleles)
        for l in range(2, 8):
            seqs = np.array(['A' * i + 'B' * (l-i) for i in range(l)])
            train_x = seq_to_one_hot(seqs, alleles=alleles)
            
            i = 0
            starting_log_lambdas = -10 * np.ones(l)
            starting_log_lambdas[i] = 0
            starting_log_lambdas = get_tensor(starting_log_lambdas)
         
            ker = SkewedVCKernel(alpha, l, q=0.7, tau=.1,
                                 starting_log_lambdas=starting_log_lambdas)
            cov1 = ker.forward(train_x, train_x).detach().numpy()
            
            ker = VCKernel(alpha, l, tau=.1,
                           starting_log_lambdas=starting_log_lambdas)
            cov2 = ker.forward(train_x, train_x).detach().numpy()
            
            mae = np.abs(cov1 - cov2).mean()
            
            axes.plot(cov1[0], label='sVC({})'.format(l))
            axes.plot(cov2[0], label='VC({})'.format(l), linestyle='--')
            assert(np.allclose(mae, 0, atol=1))
            
        axes.legend()
        axes.set(xlabel='Hamming distance', ylabel='Additive covariance')
        fig.savefig(join(TEST_DATA_DIR, 'cov_dist.png'))
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'KernelsTests']
    unittest.main()
