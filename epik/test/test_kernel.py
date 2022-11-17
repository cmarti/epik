#!/usr/bin/env python
import unittest

import numpy as np
import torch

from epik.src.kernel import SkewedVCKernel


class KernelsTests(unittest.TestCase):
    def test_calc_polynomial_coeffs(self):
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2)
        lambdas = kernel.calc_eigenvalues()
        V = torch.stack([torch.pow(lambdas, i) for i in range(3)], 1)

        B = kernel.coeffs
        P = np.array(torch.matmul(B, V))
        assert(np.allclose(P, np.eye(3), atol=1e-4))
    
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
        k1 = np.array([[1, -1, -1, 1],
                       [-1, 1, 1, -1],
                       [-1, 1, 1, -1],
                       [1, -1, -1, 1]], dtype=np.float32)
        assert(np.abs(cov - k1).mean() < 1e-4)
    
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
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'KernelsTests']
    unittest.main()
