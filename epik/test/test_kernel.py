#!/usr/bin/env python
import unittest

import numpy as np
import matplotlib.pyplot as plt
import torch

from os.path import join

from epik.src.kernel import (VarianceComponentKernel)
from epik.src.priors import LambdasExpDecayPrior, AllelesProbPrior, RhosPrior
from epik.src.utils import seq_to_one_hot, get_tensor, diploid_to_one_hot,\
    get_full_space_one_hot
from epik.src.settings import TEST_DATA_DIR
from build.lib.epik.src.kernel import ConnectednessKernel



class KernelsTests(unittest.TestCase):
    def test_calc_polynomial_coeffs(self):
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2, dtype=torch.float32)
        lambdas = get_tensor(kernel.calc_eigenvalues())
        V = torch.stack([torch.pow(lambdas, i) for i in range(3)], 1)

        B = kernel.coeffs
        P = torch.matmul(B, V).numpy()
        assert(np.allclose(P, np.eye(3), atol=1e-4))
        
    def test_vc_kernel(self):
        l, a = 2, 2
        kernel = VarianceComponentKernel(n_alleles=a, seq_length=l)
        x = get_full_space_one_hot(l, a)

        # k=0        
        lambdas = torch.tensor([1, 0, 0], dtype=torch.float32)
        cov = kernel._forward(x, x, lambdas)
        assert(np.allclose(cov, 0.25))
        
        # k=1        
        lambdas = torch.tensor([0, 1, 0], dtype=torch.float32)
        cov = kernel._forward(x, x, lambdas).numpy()
        k1 = np.array([[2, 0, 0, -2],
                       [0, 2, -2, 0],
                       [0, -2, 2, 0],
                       [-2, 0, 0, 2]], dtype=np.float32)
        assert(np.allclose(cov,  k1 / 4, atol=1e-4))
        
        # k=2
        lambdas = torch.tensor([0, 0, 1], dtype=torch.float32)
        cov = kernel._forward(x, x, lambdas).numpy()
        k2 = np.array([[1, -1, -1, 1],
                       [-1, 1, 1, -1],
                       [-1, 1, 1, -1],
                       [1, -1, -1, 1]], dtype=np.float32)
        assert(np.allclose(cov,  k2 / 4, atol=1e-4))
        
    def test_rho_kernel(self):
        l, a = 1, 2
        kernel = ConnectednessKernel(n_alleles=a, seq_length=l)
        x = get_full_space_one_hot(l, a)
        rho = torch.tensor([[0.5]])
        cov = kernel._forward(x, x, rho=rho)
        assert(cov[0, 0] == 0.75)
        assert(cov[0, 1] == 0.25)
        
        l, a = 2, 2
        kernel = ConnectednessKernel(n_alleles=a, seq_length=l)
        x = get_full_space_one_hot(l, a)
        rho = torch.tensor([[0.5, 0.5]])
        cov = kernel._forward(x, x, rho=rho)
        d0, d1, d2 = cov[0][0], cov[0][1], cov[0][3]
        assert(d1 / d0 == d2 / d1)
    
    def test_site_product_kernel_long_sequences(self):
        l, a = 100, 2
        x = (np.random.uniform(size=(2, l)) > 0.5).astype(int)
        x = np.array([''.join(y) for y in np.array(['A', 'B'])[x]])
        x = seq_to_one_hot(x, alleles=['A', 'B'])
        x = torch.tensor(x, dtype=torch.float32)
        print(x.shape)
        
        # Site product kernel
        beta = torch.tensor(np.vstack([[1, 1, 0]] * l ), dtype=torch.float32)
        theta = torch.tensor([0, -5], dtype=torch.float32)
        p_prior = AllelesProbPrior(l, a) 
        kernel = SiteProductKernel(n_alleles=a, seq_length=l, p_prior=p_prior)
        cov = kernel._forward(x, x, theta=theta, beta=beta)
        print(cov)
        
        # Generalized site product kernel
        p_prior = AllelesProbPrior(l, a, dummy_allele=False) 
        kernel = GeneralizedSiteProductKernel(n_alleles=a, seq_length=l, p_prior=p_prior)
        cov = kernel.forward(x, x)
        print(cov)
        
    def test_generalized_site_product_kernel(self):
        l, a = 1, 2
        rho_prior = RhosPrior(l, a, sites_equal=True)
        p_prior = AllelesProbPrior(l, a, dummy_allele=False)
        kernel = GeneralizedSiteProductKernel(n_alleles=a, seq_length=l,
                                              p_prior=p_prior, rho_prior=rho_prior)
        x = torch.tensor([[1, 0],
                          [0, 1]], dtype=torch.float32)
        beta = torch.tensor([[1, 1]], dtype=torch.float32)
        rho = torch.tensor([0.2], dtype=torch.float32)
        cov = kernel._forward(x, x, rho=rho, beta=beta)
        print(cov)
        
        cov = kernel.forward(x, x)
        print(cov)
         
        l, a = 2, 2
        rho_prior = RhosPrior(l, a)
        p_prior = AllelesProbPrior(l, a, dummy_allele=False) 
        kernel = GeneralizedSiteProductKernel(n_alleles=a, seq_length=l,
                                              p_prior=p_prior, rho_prior=rho_prior)
        x = torch.tensor([[1, 0, 1, 0],
                          [0, 1, 1, 0],
                          [1, 0, 0, 1],
                          [0, 1, 0, 1]], dtype=torch.float32)
        beta = torch.tensor([[1, 1],
                             [1, 1]], dtype=torch.float32)
        rho = torch.tensor([0.2, 0.2], dtype=torch.float32)
        cov = kernel._forward(x, x, rho=rho, beta=beta)
        print(cov)
         
        cov = kernel.forward(x, x)
        print(cov)
    
    def test_diploid_kernel(self):
        kernel = DiploidKernel()
        X = ['0', '1', '2']
        x = diploid_to_one_hot(X)
        
        # Purely constant function
        mu, lda, eta = 1, 0, 0
        cov = kernel._forward(x, x, mu, lda, eta).numpy()
        assert(np.allclose(cov, 1))
        
        # Purely additive function
        mu, lda, eta = 0, 1, 0
        cov = kernel._forward(x, x, mu, lda, eta).numpy()
        exp = np.array([[ 2.,  0., -2.],
                        [ 0.,  0.,  0.],
                        [-2.,  0.,  2.]])
        assert(np.allclose(cov, exp))
        
        # Purely dominant function
        mu, lda, eta = 0, 0, 1
        cov = kernel._forward(x, x, mu, lda, eta).numpy()
        exp = np.array([[ 1,  -1,   1.],
                        [-1.,  1., -1.],
                        [ 1., -1.,  1.]])
        assert(np.allclose(cov, exp))
        
    def test_skewed_vc_kernel(self):
        l, a = 2, 2
        lambdas_prior = LambdasExpDecayPrior(l, tau=0.2)
        p_prior = AllelesProbPrior(l, a)
        kernel = SkewedVCKernel(a, l, lambdas_prior, p_prior,
                                dtype=torch.float32)
        
        logp = torch.tensor([[0, 0, -10], [0, 0, -10]], dtype=torch.float32)
        logp = p_prior.normalize_logp(logp)
        x = torch.tensor([[1, 0, 1, 0],
                          [0, 1, 1, 0],
                          [1, 0, 0, 1],
                          [0, 1, 0, 1]], dtype=torch.float32)

        # k=0        
        lambdas = torch.tensor([1, 0, 0], dtype=torch.float32)
        cov = kernel._forward(x, x, lambdas, logp)
        assert(np.allclose(cov, 1))
         
        # k=1        
        lambdas = torch.tensor([0, 1, 0], dtype=torch.float32)
        cov = kernel._forward(x, x, lambdas, logp).numpy()
        k1 = np.array([[2, 0, 0, -2],
                       [0, 2, -2, 0],
                       [0, -2, 2, 0],
                       [-2, 0, 0, 2]], dtype=np.float32)
        assert(np.abs(cov - k1).mean() < 1e-4)
         
        # k=2
        lambdas = torch.tensor([0, 0, 1], dtype=torch.float32)
        cov = kernel._forward(x, x, lambdas, logp).numpy()
        k2 = np.array([[1, -1, -1, 1],
                       [-1, 1, 1, -1],
                       [-1, 1, 1, -1],
                       [1, -1, -1, 1]], dtype=np.float32)
        assert(np.abs(cov - k2).mean() < 1e-4)
    
        # Longer seqs
        l, a = 6, 2
        lambdas_prior = LambdasExpDecayPrior(l, tau=0.2)
        p_prior = AllelesProbPrior(l, a)
        kernel = SkewedVCKernel(a, l, lambdas_prior, p_prior,
                                dtype=torch.float32)
        
        logp = torch.tensor([[0, 0, -10]] * l, dtype=torch.float32)
        logp = p_prior.normalize_logp(logp)
        x = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                          [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                          [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
                          [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
                          [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
                          [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
                          [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]], dtype=torch.float32)
        
        lambdas = torch.tensor([0, 1, 0, 0, 0, 0, 0], dtype=torch.float32)
        cov = kernel._forward(x[:1], x[1:], lambdas, logp).numpy()
        assert(np.allclose(cov[0][:-1] - cov[0][1:], 2, atol=0.05))
        
        # More than 2 alleles
        l, a = 2, 3
        lambdas_prior = LambdasExpDecayPrior(l, tau=0.2)
        p_prior = AllelesProbPrior(l, a)
        kernel = SkewedVCKernel(a, l, lambdas_prior, p_prior,
                                dtype=torch.float32)
        logp = torch.tensor([[0, 0, 0, -10]] * l, dtype=torch.float32)
        logp = p_prior.normalize_logp(logp)
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
        cov = kernel._forward(x[:1], x, lambdas, logp).numpy()
        k1 = np.array([4, 1, 1, 1, -2, -2, 1, -2, -2])
        assert(np.abs(cov[0] - k1).mean() < 1e-4)
    
    def test_skewed_vc_kernel_different_qs(self):
        for q in np.linspace(0.1, 0.9, 11):
            kernel = SkewedVCKernel(n_alleles=2, seq_length=2, q=q, dtype=torch.float32)
            logp = torch.log(torch.tensor([[0.5, 0.5, 0.00001],
                                            [0.5, 0.5, 0.00001]], dtype=torch.float32))
            x = torch.tensor([[1, 0, 1, 0],
                              [0, 1, 1, 0],
                              [1, 0, 0, 1],
                              [0, 1, 0, 1]], dtype=torch.float32)
    
            # k=0        
            lambdas = torch.tensor([1, 0, 0], dtype=torch.float32)
            cov = kernel._forward(x, x, lambdas, logp).numpy()
            assert(np.allclose(cov, 1, atol=1e-4))
            
            # k=1        
            lambdas = torch.tensor([0, 1, 0], dtype=torch.float32)
            cov = kernel._forward(x, x, lambdas, logp).numpy()
            k1 = np.array([[2, 0, 0, -2],
                           [0, 2, -2, 0],
                           [0, -2, 2, 0],
                           [-2, 0, 0, 2]], dtype=np.float32)
            assert(np.abs(cov - k1).mean() < 1e-4)
            
            # k=2
            lambdas = torch.tensor([0, 0, 1], dtype=torch.float32)
            cov = kernel._forward(x, x, lambdas, logp).numpy()
            k2 = np.array([[1, -1, -1, 1],
                           [-1, 1, 1, -1],
                           [-1, 1, 1, -1],
                           [1, -1, -1, 1]], dtype=np.float32)
            assert(np.abs(cov - k2).mean() < 1e-4)
    
    def test_skewed_vc_kernel_gpu(self):
        gpu = torch.device('cuda:0')
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2, dtype=torch.float32).to(gpu)
        logp = torch.log(torch.tensor([[0.5, 0.5, 0.00001],
                                        [0.5, 0.5, 0.00001]], dtype=torch.float32)).to(gpu)
        x = torch.tensor([[1, 0, 1, 0],
                          [0, 1, 1, 0],
                          [1, 0, 0, 1],
                          [0, 1, 0, 1]], dtype=torch.float32)
        xgpu = x.to(gpu)

        # k=0        
        lambdas = torch.tensor([1, 0, 0], dtype=torch.float32).to(gpu)
        cov = kernel._forward(xgpu, xgpu, lambdas, logp).cpu().numpy()
        assert(np.allclose(cov, 1))
        
        # k=1        
        lambdas = torch.tensor([0, 1, 0], dtype=torch.float32).to(gpu)
        cov = kernel._forward(xgpu, xgpu, lambdas, logp).cpu().numpy()
        k1 = np.array([[2, 0, 0, -2],
                       [0, 2, -2, 0],
                       [0, -2, 2, 0],
                       [-2, 0, 0, 2]], dtype=np.float32)
        assert(np.abs(cov - k1).mean() < 1e-4)
        
        # k=2
        lambdas = torch.tensor([0, 0, 1], dtype=torch.float32).to(gpu)
        cov = kernel._forward(xgpu, xgpu, lambdas, logp).cpu().numpy()
        k2 = np.array([[1, -1, -1, 1],
                       [-1, 1, 1, -1],
                       [-1, 1, 1, -1],
                       [1, -1, -1, 1]], dtype=np.float32)
        assert(np.abs(cov - k2).mean() < 1e-4)
        
        # Test now with the public method
        logp = torch.log(torch.tensor([[0.5, 0.5, 0.00001],
                                        [0.5, 0.5, 0.00001]], dtype=torch.float32))
        log_lambdas = torch.tensor([0, -10], dtype=torch.float32)

        for q in np.linspace(0.1, 0.9, 11):
            kernel = SkewedVCKernel(n_alleles=2, seq_length=2, q=q,
                                    log_lambdas0=log_lambdas,
                                    starting_p=torch.exp(logp),
                                    dtype=torch.float32).to(gpu)
            cov_gpu = kernel.forward(xgpu, xgpu).detach().cpu().numpy()
            
            kernel = SkewedVCKernel(n_alleles=2, seq_length=2, q=q,
                                    log_lambdas0=log_lambdas,
                                    starting_p=torch.exp(logp),
                                    dtype=torch.float32)
            cov_cpu = kernel.forward(x, x).detach().numpy()
            assert(np.allclose(cov_gpu, cov_cpu, atol=1e-3))
            assert(np.allclose(cov_gpu, k1, atol=1e-3))
        
    def test_skewed_vc_kernel_diag(self):
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2, dtype=torch.float32)
        logp = torch.log(torch.tensor([[0.45, 0.45, 0.1],
                                        [0.45, 0.45, 0.1]], dtype=torch.float32))
        x = torch.tensor([[1, 0, 1, 0],
                          [0, 1, 1, 0],
                          [1, 0, 0, 1],
                          [0, 1, 0, 1]], dtype=torch.float32)

        lambdas = torch.tensor([1, 0.5, 0.1], dtype=torch.float32)
        cov = kernel._forward(x, x, lambdas, logp, diag=False)
        var = kernel._forward(x, x, lambdas, logp, diag=True)
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
    import sys;sys.argv = ['', 'KernelsTests.test_rho_kernel']
    unittest.main()
