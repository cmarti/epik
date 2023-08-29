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
    rho0 = torch.tensor([0.9, 0.7, 0.5, 0.1, 0.3])
    alleles = ['A', 'C', 'G', 'T']
    l, a = rho0.shape[0], len(alleles)
    sigma2 = 1e-4
    reps = 5
    n = 50

    seqs = np.array([''.join(gt) for gt in product(alleles, repeat=l)])
    X = seq_to_one_hot(seqs, alleles)
    model = GPmodel(l, a, rho0=rho0, track_progress=True, sites_equal=False)
    y_true, y, y_var = model.simulate(X, sigma2)
    
    splits = split_training_test(X, y, y_var, ptrain=0.9)
    train_x, train_y, test_x, test_y, train_y_var = splits

    params = model.train(train_x, train_y, train_y_var, n=n)
    test_y_pred = model.predict(train_x, train_y, train_y_var, test_x)
    record = model.evaluate_predictions(test_y, test_y_pred)
    rhos = np.vstack([x['rho'].detach().numpy() for x in model.history])
    rho = pd.DataFrame(rhos, columns=['rho{}'.format(i) for i in range(1, l+1)])
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
    lim = (1e-2, 2e0)
    axes.set(xlabel=r'$\rho_{Sim}$', ylabel=r'$\rho_{Inferred}$',
             yscale='log', xscale='log', aspect='equal',
             xlim=lim, ylim=lim)
    axes.plot(lim, lim, lw=0.5, c='grey', linestyle='--')
    axes.grid(alpha=0.5)
    
    fig.tight_layout()
    fig.savefig('rhop.inference.png', dpi=300)
    
