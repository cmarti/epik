#!/usr/bin/env python
import pandas as pd
import numpy as np
import torch

import seaborn as sns
import matplotlib.pyplot as plt

from itertools import product

from epik.src.utils import seq_to_one_hot, split_training_test
from epik.analysis.simulation_global_epistasis2 import GPmodel
from sklearn.decomposition._pca import PCA


if __name__ == '__main__':
    alleles = ['A', 'C', 'G', 'T']
    l, a = 5, len(alleles)
    sigma2 = 1e-4
    reps = 5
    rho0 = torch.tensor([0.01, 0.02, 0.05, 0.1, 0.2])
    n = 100

    seqs = np.array([''.join(gt) for gt in product(alleles, repeat=l)])
    X = seq_to_one_hot(seqs, alleles)
    model = GPmodel(l, a, rho0=rho0, track_progress=True, sites_equal=False)
    y_true, y, y_var = model.simulate(X, sigma2)
    
    splits = split_training_test(X, y, y_var, ptrain=0.5)
    train_x, train_y, test_x, test_y, train_y_var = splits

    params = model.train(train_x, train_y, train_y_var, n=n)
    test_y_pred = model.predict(train_x, train_y, train_y_var, test_x)
    record = model.evaluate_predictions(test_y, test_y_pred)
    rhos = np.vstack([x['rho'].detach().numpy() for x in model.history])
    rho = pd.DataFrame(rhos, columns=['rho{}'.format(i) for i in range(1, 6)])
    h = pd.DataFrame({'i': np.arange(1, n+1),
                      'loss': [x['loss'] for x in model.history]})
    h = pd.concat([rho, h], axis=1)
    
    rho0 = rho0.numpy()
    rho1 = params['rho']
    
    fig, subplots = plt.subplots(1, 2, figsize=(2 * 3.5, 1 * 3.5))
    
    axes = subplots[0]
    m = PCA(n_components=2)
    X = m.fit_transform(np.log(rhos))
    axes.scatter(x=X[:, 0], y=X[:, 1], c=-h['loss'], zorder=2, s=25, lw=0.5,
                 edgecolor='black')
    axes.plot(X[:, 0], X[:, 1], c='black', zorder=1)
    axes.set(xlabel='PC1', ylabel='PC2')
    
    axes = subplots[1]
    axes.scatter(rho0, rho1)
    axes.set(xlabel=r'$\rho_{Sim}$', ylabel=r'$\rho_{Inferred}$',
             yscale='log', xscale='log', aspect='equal',
             xlim=(0.001, 0.5), ylim=(0.001, 0.5))
    axes.plot((0.001, 0.5), (0.001, 0.5), lw=0.5, c='grey', linestyle='--')
    axes.grid(alpha=0.5)
    axes.legend(loc=2)
    
    fig.tight_layout()
    fig.savefig('rhop.inference.png', dpi=300)
    
