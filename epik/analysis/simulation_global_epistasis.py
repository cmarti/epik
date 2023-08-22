#!/usr/bin/env python
import unittest
import sys

import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import product
from scipy.stats import pearsonr

from epik.src.utils import seq_to_one_hot, split_training_test
from epik.src.model import EpiK
from epik.src.kernel import GeneralizedSiteProductKernel
from epik.src.priors import AllelesProbPrior, RhosPrior


def sample_params(l, a):
    rho0 = np.random.uniform(size=l)
    p0 = np.vstack([np.random.dirichlet(np.ones(a)) for _ in range(l)])
    beta0 = -np.log(p0 / (1 - p0))
    return(torch.tensor(rho0), torch.tensor(beta0))


def get_model(l, a, rho0=None, beta0=None, train_p=True):
    rhos_prior = RhosPrior(seq_length=l, n_alleles=a,
                           rho0=rho0, sites_equal=False, train=True)
    p_prior = AllelesProbPrior(seq_length=l, n_alleles=a, dummy_allele=False,
                               beta0=beta0, sites_equal=False, train=train_p)
    kernel = GeneralizedSiteProductKernel(n_alleles=a, seq_length=l,
                                          rho_prior=rhos_prior, p_prior=p_prior)
    model = EpiK(kernel, likelihood_type='Gaussian', track_progress=False)
    return(model)


def simulate_data(model, c=1, sigma2=0.005):
    alleles = ['A', 'C', 'T', 'G']
    seqs = np.array([''.join(gt) for gt in product(alleles, repeat=model.kernel.l)])
    X = seq_to_one_hot(seqs, alleles)
    phi = model.sample(X, n=1, sigma2=1e-4).flatten()
    phi =  (phi - phi.mean()) / phi.std()
    y_true = torch.exp(c*phi) / (1 + torch.exp(c*phi))
    y = torch.normal(y_true, np.sqrt(sigma2))
    y_var = sigma2 * torch.ones(X.shape[0])
    return(X, phi, y_true, y, y_var)


def split_data(X, y, y_var, p=0.8):
    return(split_training_test(X, y, y_var, ptrain=p))


def predict(model, train_x, train_y, train_y_var, test_x):
    model.set_data(train_x, train_y, train_y_var)
    ypred = model.predict(test_x).detach()
    return(ypred)


def train(model, train_x, train_y, train_y_var, n=100):
    model.set_data(train_x, train_y, train_y_var)
    model.fit(n_iter=n)
    params = model.kernel.get_params()
    params = {k: v.detach().numpy() for k, v in params.items()}
    return(params)


def evaluate_predictions(p, ypred, test_y):
    ypred, test_y = ypred.numpy(), test_y.numpy()
    r2 = pearsonr(ypred, test_y)[0] ** 2
    mse = np.mean((ypred - test_y) ** 2)
    return({'p': p, 'r2': r2, 'mse': mse})


def evaluate_params(params0, params, p):
    logrho0 = np.log(params0['rho'])
    logrho = np.log(params['rho'])
    rho_r = pearsonr(logrho, logrho0)[0]
    
    beta0 = params0['beta'].flatten()
    beta = params['beta'].flatten()
    beta_r = pearsonr(beta, beta0)[0]
    
    return({'p': p, 'rho_r': rho_r, 'beta_r': beta_r})


def plot_global_epistasis(phi, y, c, ytrue, ypred1, ypred2):
    fig, subplots = plt.subplots(1, 3, figsize=(12, 3.5))

    # Plot real values
    xs = np.linspace(-5, 5, 101)
    ys = np.exp(xs*c) / (1+np.exp(xs*c))
    axes = subplots[0]
    axes.scatter(phi, y, alpha=0.2, s=10)
    axes.plot(xs, ys, c='orange')
    axes.set(xlabel=r'Latent phenotype $\phi$ ($\rho$ kernel)',
             ylabel=r'Measurement $y$',
             xlim=(1.1 * phi.min(), 1.1*phi.max()),
             title='Simulated data')
    axes.grid(alpha=0.1)
    
    # Plot inferred y values 2
    axes = subplots[1]
    axes.scatter(ytrue, ypred1, alpha=0.2, s=10)
    axes.set(xlabel=r'Latent phenotype $\phi$ ($\rho$ kernel)',
             ylabel=r'Measurement $y$',
             title=r'$\rho$ kernel')
    axes.grid(alpha=0.1)
    
    # Plot inferred y values 2
    axes = subplots[2]
    axes.scatter(ytrue, ypred2, alpha=0.2, s=10)
    axes.set(xlabel=r'Latent phenotype $\phi(\rho\ kernel)$',
             ylabel=r'Measurement $y$',
             title=r'$\rho\pi$ kernel')
    axes.grid(alpha=0.1)
    
    fig.tight_layout()
    fig.savefig('global_epistasis.png', dpi=300)


if __name__ == '__main__':
    l, a = 5, 4
    sigma2 = 0.005
    reps = 10
    c = 5

    # Sample parameters
    np.random.seed(1)
    rho0, _ = sample_params(l, a)
    
    # Simulate from rho kernel with non-linearity
    model0 = get_model(l, a, rho0)
    params0 = {k: v.detach().numpy() for k, v in model0.kernel.get_params().items()}
    
    results = []
    for _ in range(reps):
        
        # Generate data
        X, phi, y_true, y, y_var = simulate_data(model0, c=c, sigma2=sigma2)   
            
        for i in range(1, 10):
            p = i/10.            
            train_x, train_y, test_x, test_y, train_y_var = split_data(X, y, y_var, p=p)
            
            # Train only rho
            model = get_model(l, a, rho0, train_p=False)        
            params = train(model, train_x, train_y, train_y_var)
            ypred1 = predict(model0, train_x, train_y, train_y_var, test_x)
            record = evaluate_predictions(p, ypred1, test_y)
            record['label'] = 'Rho kernel'
            record.update(evaluate_params(params0, params, p))
            results.append(record)
            print(record)
            
            # Train p
            model = get_model(l, a, rho0, train_p=True)        
            params = train(model, train_x, train_y, train_y_var)
            ypred2 = predict(model, train_x, train_y, train_y_var, test_x)
            record = evaluate_predictions(p, ypred2, test_y)
            record['label'] = 'Pi kernel'
            record.update(evaluate_params(params0, params, p))
            results.append(record)
            print(record)
            
            plot_global_epistasis(phi, y, c, test_y, ypred1, ypred2)
    
    results = pd.DataFrame(results)
    results.to_csv('predictions.global_epistasis.csv')
