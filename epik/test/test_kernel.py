#!/usr/bin/env python
import unittest

import numpy as np
import matplotlib.pyplot as plt
import torch

from epik.src.kernel import SkewedVCKernel,  VCKernel, ExponentialKernel,\
    SiteProductKernel
from epik.src.utils import seq_to_one_hot, get_tensor
from epik.src.settings import TEST_DATA_DIR
from os.path import join


class KernelsTests(unittest.TestCase):
    def test_calc_polynomial_coeffs(self):
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2, dtype=torch.float32)
        lambdas = get_tensor(kernel.calc_eigenvalues())
        V = torch.stack([torch.pow(lambdas, i) for i in range(3)], 1)

        B = kernel.coeffs
        P = torch.matmul(B, V).numpy()
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
        
    def test_site_product_kernel(self):
        kernel = SiteProductKernel(n_alleles=2, seq_length=1)
        x = torch.tensor([[1, 0],
                          [0, 1]], dtype=torch.float32)
        w = torch.tensor([0], dtype=torch.float32)
        beta = torch.tensor([1], dtype=torch.float32)
        a = torch.tensor([2], dtype=torch.float32)
        cov = kernel._forward(x, x, a=a, beta=beta, w=w)
        print(cov)
        
        kernel = SiteProductKernel(n_alleles=2, seq_length=2)
        x = torch.tensor([[1, 0, 1, 0],
                          [0, 1, 1, 0],
                          [1, 0, 0, 1],
                          [0, 1, 0, 1]], dtype=torch.float32)
        w = torch.tensor([0, 0], dtype=torch.float32)
        beta = torch.tensor([1], dtype=torch.float32)
        a = torch.tensor([2], dtype=torch.float32)
        cov = kernel._forward(x, x, a=a, beta=beta, w=w)
        print(cov)

    
    def test_exponential_kernel(self):
        kernel = ExponentialKernel(n_alleles=2, seq_length=2, 
                                   starting_lengthscale=1)
        x = torch.tensor([[1, 0, 1, 0],
                          [0, 1, 1, 0],
                          [1, 0, 0, 1],
                          [0, 1, 0, 1]], dtype=torch.float32)
        cov = kernel.forward(x, x).detach().numpy()
        k = np.array([[1., 0.5, 0.5, 0.25],
                      [0.5, 1., 0.25, 0.5],
                      [0.5, 0.25, 1., 0.5],
                      [0.25, 0.5, 0.5, 1.]])
        assert(np.allclose(cov, k, atol=1e-3))
        
        # Different length scale
        kernel = ExponentialKernel(n_alleles=2, seq_length=2, 
                                   starting_lengthscale=2)
        cov = kernel.forward(x, x).detach().numpy()
        k = np.array([[1., 0.25, 0.25, 0.0625],
                      [0.25, 1., 0.0625, 0.25],
                      [0.25, 0.0625, 1., 0.25],
                      [0.0625, 0.25, 0.25, 1.]])
        assert(np.allclose(cov, k, atol=1e-3))
    
        # Increasing length    
        kernel = ExponentialKernel(n_alleles=2, seq_length=3, 
                                   starting_lengthscale=1)
        x = torch.tensor([[1, 0, 1, 0, 1, 0],
                          [0, 1, 1, 0, 1, 0],
                          [1, 0, 0, 1, 1, 0],
                          [0, 1, 0, 1, 1, 0],
                          [1, 0, 1, 0, 0, 1],
                          [0, 1, 1, 0, 0, 1],
                          [1, 0, 0, 1, 0, 1],
                          [0, 1, 0, 1, 0, 1]], dtype=torch.float32)
        cov = kernel.forward(x, x).detach().numpy()
        k0 = np.array([1., 0.5, 0.5, 0.25, 0.5, 0.25, 0.25, 0.125])
        assert(np.allclose(cov[0], k0, atol=1e-3))
        
        # Increasing alleles
        kernel = ExponentialKernel(n_alleles=3, seq_length=2, 
                                   starting_lengthscale=1)
        x = torch.tensor([[1, 0, 0, 1, 0, 0],
                          [0, 1, 0, 1, 0, 0],
                          [0, 0, 1, 1, 0, 0],
                          [1, 0, 0, 0, 1, 0],
                          [0, 1, 0, 0, 1, 0],
                          [0, 0, 1, 0, 1, 0],
                          [1, 0, 0, 0, 0, 1],
                          [0, 1, 0, 0, 0, 1],
                          [0, 0, 1, 0, 0, 1]], dtype=torch.float32)
        cov = kernel.forward(x, x).detach().numpy()
        k0 = np.array([1., 0.5, 0.5,
                       0.5, 0.25, 0.25,
                       0.5, 0.25, 0.25])
        assert(np.allclose(cov[0], k0, atol=1e-3))
        
    def test_skewed_vc_lambdas_parametrizations(self):
        x = torch.tensor([[1, 0, 1, 0],
                          [0, 1, 1, 0],
                          [1, 0, 0, 1],
                          [0, 1, 0, 1]], dtype=torch.float32)
        
        # Constant component
        lambdas = torch.tensor([1, 0, 0], dtype=torch.float32)
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2, dtype=torch.float32,
                                lambdas_prior='monotonic_decay')
        cov = kernel._forward(x, x, lambdas, kernel.log_p)
        assert(np.allclose(cov.detach().numpy(), 1))
        
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2, dtype=torch.float32,
                                lambdas_prior='2nd_order_diff')
        cov = kernel._forward(x, x, lambdas, kernel.log_p)
        assert(np.allclose(cov.detach().numpy(), 1))
        
        # Arbitrary lambdas
        lambdas = torch.tensor([1, 0.2, 0.05], dtype=torch.float32)
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2, dtype=torch.float32,
                                lambdas_prior='monotonic_decay')
        cov1 = kernel._forward(x, x, lambdas, kernel.log_p).detach().numpy()
        
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2, dtype=torch.float32,
                                lambdas_prior='2nd_order_diff')
        cov2 = kernel._forward(x, x, lambdas, kernel.log_p).detach().numpy()
        assert(np.allclose(cov1, cov2))
        
        # With initialized lambdas to ensure transformation to the right thetas
        log_lambdas0 = torch.tensor([0, -10], dtype=torch.float32)
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2, dtype=torch.float32,
                                lambdas_prior='monotonic_decay', log_lambdas0=log_lambdas0)
        cov1 = kernel.forward(x, x).detach().numpy()
        
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2, dtype=torch.float32,
                                lambdas_prior='2nd_order_diff', log_lambdas0=log_lambdas0)
        cov2 = kernel.forward(x, x).detach().numpy()
        assert(np.allclose(cov1, cov2))
    
    def test_skewed_vc_kernel(self):
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2, dtype=torch.float32)
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
        kernel = SkewedVCKernel(n_alleles=2, seq_length=5, q=0.7, dtype=torch.float32)
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
        kernel = SkewedVCKernel(n_alleles=3, seq_length=2, dtype=torch.float32)
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
    
    def test_skewed_vc_kernel_different_qs(self):
        for q in np.linspace(0.1, 0.9, 11):
            kernel = SkewedVCKernel(n_alleles=2, seq_length=2, q=q, dtype=torch.float32)
            log_p = torch.log(torch.tensor([[0.5, 0.5, 0.00001],
                                            [0.5, 0.5, 0.00001]], dtype=torch.float32))
            x = torch.tensor([[1, 0, 1, 0],
                              [0, 1, 1, 0],
                              [1, 0, 0, 1],
                              [0, 1, 0, 1]], dtype=torch.float32)
    
            # k=0        
            lambdas = torch.tensor([1, 0, 0], dtype=torch.float32)
            cov = kernel._forward(x, x, lambdas, log_p).numpy()
            assert(np.allclose(cov, 1, atol=1e-4))
            
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
    
    def test_skewed_vc_kernel_gpu(self):
        gpu = torch.device('cuda:0')
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2, dtype=torch.float32).to(gpu)
        log_p = torch.log(torch.tensor([[0.5, 0.5, 0.00001],
                                        [0.5, 0.5, 0.00001]], dtype=torch.float32)).to(gpu)
        x = torch.tensor([[1, 0, 1, 0],
                          [0, 1, 1, 0],
                          [1, 0, 0, 1],
                          [0, 1, 0, 1]], dtype=torch.float32)
        xgpu = x.to(gpu)

        # k=0        
        lambdas = torch.tensor([1, 0, 0], dtype=torch.float32).to(gpu)
        cov = kernel._forward(xgpu, xgpu, lambdas, log_p).cpu().numpy()
        assert(np.allclose(cov, 1))
        
        # k=1        
        lambdas = torch.tensor([0, 1, 0], dtype=torch.float32).to(gpu)
        cov = kernel._forward(xgpu, xgpu, lambdas, log_p).cpu().numpy()
        k1 = np.array([[2, 0, 0, -2],
                       [0, 2, -2, 0],
                       [0, -2, 2, 0],
                       [-2, 0, 0, 2]], dtype=np.float32)
        assert(np.abs(cov - k1).mean() < 1e-4)
        
        # k=2
        lambdas = torch.tensor([0, 0, 1], dtype=torch.float32).to(gpu)
        cov = kernel._forward(xgpu, xgpu, lambdas, log_p).cpu().numpy()
        k2 = np.array([[1, -1, -1, 1],
                       [-1, 1, 1, -1],
                       [-1, 1, 1, -1],
                       [1, -1, -1, 1]], dtype=np.float32)
        assert(np.abs(cov - k2).mean() < 1e-4)
        
        # Test now with the public method
        log_p = torch.log(torch.tensor([[0.5, 0.5, 0.00001],
                                        [0.5, 0.5, 0.00001]], dtype=torch.float32))
        log_lambdas = torch.tensor([0, -10], dtype=torch.float32)

        for q in np.linspace(0.1, 0.9, 11):
            kernel = SkewedVCKernel(n_alleles=2, seq_length=2, q=q,
                                    log_lambdas0=log_lambdas,
                                    starting_p=torch.exp(log_p),
                                    dtype=torch.float32).to(gpu)
            cov_gpu = kernel.forward(xgpu, xgpu).detach().cpu().numpy()
            
            kernel = SkewedVCKernel(n_alleles=2, seq_length=2, q=q,
                                    log_lambdas0=log_lambdas,
                                    starting_p=torch.exp(log_p),
                                    dtype=torch.float32)
            cov_cpu = kernel.forward(x, x).detach().numpy()
            assert(np.allclose(cov_gpu, cov_cpu, atol=1e-3))
            assert(np.allclose(cov_gpu, k1, atol=1e-3))
        
    def test_skewed_vc_kernel_diag(self):
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2, dtype=torch.float32)
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
                                log_lambdas0=np.array([0, -10]),
                                dtype=torch.float32)
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
                    log_lambdas0 = -10 * np.ones(l)
                    log_lambdas0[i] = 0
                 
                    ker = VCKernel(alpha, l, tau=.1,
                                   log_lambdas0=log_lambdas0)
                    cov1 = ker.forward(train_x, train_x).detach().numpy()
                    
                    ker = SkewedVCKernel(alpha, l, q=0.7, tau=.1,
                                         log_lambdas0=log_lambdas0,
                                         dtype=torch.float32,
                                         lambdas_prior='2nd_order_diff')                    
                    cov2 = ker.forward(train_x, train_x).detach().numpy()
                    
                    # Tests they are all similar
                    logfc = np.nanmean(np.log2(cov1 / cov2))
                    assert(np.allclose(logfc, 0, atol=1))
                    
    def test_vc_kernels_variable_lengths_gpu(self):
        gpu = torch.device('cuda:0')
        for alleles in [['A', 'B'], ['A', 'B', 'C']]:
            alpha = len(alleles)
            for l in range(2, 8):
                seqs = np.array(['A' * i + 'B' * (l-i) for i in range(l)])
                train_x = seq_to_one_hot(seqs, alleles=alleles)
                train_x_gpu = train_x.to(gpu)
                
                for i in range(l):
                    log_lambdas0 = -10 * np.ones(l)
                    log_lambdas0[i] = 0
                    log_lambdas0 = get_tensor(log_lambdas0)
                 
                    # CPU
                    ker = VCKernel(alpha, l, tau=.1,
                                   log_lambdas0=log_lambdas0)
                    cov1 = ker.forward(train_x, train_x).detach().numpy()
                    
                    ker = SkewedVCKernel(alpha, l, q=0.7, tau=.1,
                                         log_lambdas0=log_lambdas0)                    
                    cov2 = ker.forward(train_x, train_x).detach().numpy()
                    
                    # GPU
                    ker = VCKernel(alpha, l, tau=.1,
                                   log_lambdas0=log_lambdas0).to(gpu)
                    cov3 = ker.forward(train_x_gpu, train_x_gpu).detach().cpu().numpy()
                     
                    ker = SkewedVCKernel(alpha, l, tau=.1, q=0.7,
                                         log_lambdas0=log_lambdas0).to(gpu)
                    cov4 = ker.forward(train_x_gpu, train_x_gpu).detach().cpu().numpy()
                    
                    # Tests they are all similar
                    logfc = np.nanmean(np.log2(cov1 / cov2))
                    assert(np.allclose(logfc, 0, atol=1e-2))
                    
                    logfc = np.nanmean(np.log2(cov1 / cov3))
                    assert(np.allclose(logfc, 0, atol=1e-2))
                     
                    logfc = np.nanmean(np.log2(cov1 / cov4))
                    assert(np.allclose(logfc, 0, atol=1e-2))

    def test_skewed_vc_kernel_exp_decay(self):
        alleles = ['A', 'B']
        alpha = len(alleles)
        l = 12
        
        seqs = np.array(['A' * i + 'B' * (l-i) for i in range(l)])
        train_x = seq_to_one_hot(seqs, alleles=alleles).to(torch.float64)
        log_lambdas0 = -np.arange(l).astype(float)
     
        ker = VCKernel(alpha, l, tau=.1,
                       log_lambdas0=log_lambdas0)
        cov1 = ker.forward(train_x[:1, :], train_x).detach().numpy()
        
        ker = SkewedVCKernel(alpha, l, q=0.7, tau=.1,
                             log_lambdas0=log_lambdas0)                    
        cov2 = ker.forward(train_x[:1, :], train_x).detach().numpy()
        
        logfc = np.nanmean(np.log2(cov1 / cov2))
        assert(np.allclose(logfc, 0, atol=1e-2))
                    
    def test_plot_vc_cov_d_function(self):
        fig, axes = plt.subplots(1, 1, figsize=(5, 4))
        
        alleles = ['A', 'B']
        alpha = len(alleles)
        for l in range(2, 8):
            seqs = np.array(['A' * i + 'B' * (l-i) for i in range(l)])
            train_x = seq_to_one_hot(seqs, alleles=alleles).to(torch.float64)
            
            i = 0
            log_lambdas0 = -10 * np.ones(l)
            log_lambdas0[i] = 0
         
            ker = SkewedVCKernel(alpha, l, q=0.7, tau=.1,
                                 log_lambdas0=log_lambdas0,
                                 dtype=torch.float64)
            cov1 = ker.forward(train_x, train_x).detach().numpy()
            
            ker = VCKernel(alpha, l, tau=.1,
                           log_lambdas0=log_lambdas0)
            cov2 = ker.forward(train_x, train_x).detach().numpy()
            
            axes.plot(cov1[0], label='sVC({})'.format(l))
            axes.plot(cov2[0], label='VC({})'.format(l), linestyle='--')
            
            logfc = np.nanmean(np.log2(cov1 / cov2))
            assert(np.allclose(logfc, 0, atol=1e-2))
            
        axes.legend()
        axes.set(xlabel='Hamming distance', ylabel='Additive covariance')
        fig.savefig(join(TEST_DATA_DIR, 'cov_dist.png'))
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'KernelsTests.test_site_product_kernel']
    unittest.main()
