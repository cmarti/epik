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
        assert(np.allclose(B.sum(0), [1, 0, 0]))
        assert(np.allclose(B.sum(1), [1, 0, 0]))
        
        P = np.array(torch.matmul(B, V))
        assert(np.allclose(P, np.eye(3), atol=1e-4))
        
    # def test_odds_ratio(self):
    #     l = 8
    #     x = np.arange(l+1)
    #     q = (l - 1) / l
    #     fx = q ** x / (1 - q) ** x
    #     print(x)
    #     print(fx)
    
    
    def test_skewed_vc_kernel(self):
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2)
        
        x = torch.tensor([[1, 0, 1, 0],
                          [0, 1, 1, 0],
                          [1, 0, 0, 1],
                          [0, 1, 0, 1]], dtype=torch.float32)
        lambdas = torch.tensor([1, 0, 0], dtype=torch.float32)
        log_p = torch.log(torch.tensor([[0.5, 0.5],
                                        [0.5, 0.5]], dtype=torch.float32))
        cov = kernel(x, x, lambdas, log_p)
        print(cov)
        assert(np.allclose(cov, 1))
        
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'KernelsTests.test_skewed_vc_kernel']
    unittest.main()
