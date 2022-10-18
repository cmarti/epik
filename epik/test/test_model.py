#!/usr/bin/env python
import unittest

import pandas as pd
import numpy as np
import torch

from os.path import join

from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood

from epik.src.kernel import SkewedVCKernel
from epik.src.settings import TEST_DATA_DIR
from epik.src.model import EpiK



class ModelsTests(unittest.TestCase):
    def test_epik(self):
        data = pd.read_csv(join(TEST_DATA_DIR, 'smn1data.csv'),
                           header=None, index_col=0, names=['m', 'std'])
        
        kernel = SkewedVCKernel(n_alleles=2, seq_length=2)
        likelihood = GaussianLikelihood()
        
        model = EpiK(kernel, likelihood)
        
        X = np.array(['AA', 'AB', 'BA', 'BB'])
        y = torch.tensor([0.2, 1.1, 0.5, 1.5])
        
        # Ensure proper 1-hot encoding
        x = model.seq_to_one_hot(X).numpy()
        onehot = np.array([[1, 0, 1, 0],
                           [1, 0, 0, 1],
                           [0, 1, 1, 0],
                           [0, 1, 0, 1.]])
        assert(np.allclose(x - onehot, 0))

        with torch.autograd.set_detect_anomaly(True):
            model.fit(X, y)
        
        
        
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'ModelsTests']
    unittest.main()
