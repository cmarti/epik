#!/usr/bin/env python
import unittest
import sys

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from scipy.stats import pearsonr

from epik.src.utils import seq_to_one_hot, split_training_test
from epik.src.model import EpiK
from epik.src.kernel import GeneralizedSiteProductKernel
from epik.old.priors import AllelesProbPrior, RhosPrior


def sample_params(l, a):
    rho0 = np.random.uniform(size=l)
    p0 = np.vstack([np.random.dirichlet(np.ones(a)) for _ in range(l)])
    beta0 = -np.log(p0 / (1 - p0))
    return(torch.tensor(rho0), torch.tensor(beta0))


def get_model(l, a, rho0=None, beta0=None):
    rhos_prior = RhosPrior(seq_length=l, n_alleles=a,
                           rho0=rho0, sites_equal=False, train=True)
    p_prior = AllelesProbPrior(seq_length=l, n_alleles=a, dummy_allele=False,
                               beta0=beta0, sites_equal=False, train=True)
    kernel = GeneralizedSiteProductKernel(n_alleles=a, seq_length=l,
                                          rho_prior=rhos_prior, p_prior=p_prior)
    model = EpiK(kernel, likelihood_type='Gaussian', track_progress=False)
    return(model)


def simulate_data(model, sigma2=0.005):
    alleles = ['A', 'C', 'T', 'G']
    seqs = np.array([''.join(gt) for gt in product(alleles, repeat=model.kernel.l)])
    X = seq_to_one_hot(seqs, alleles)
    y = model.sample(X, n=1, sigma2=sigma2).flatten()
    y_var = sigma2 * torch.ones(X.shape[0])
    return(X, y, y_var)


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

if __name__ == '__main__':
    l, a = 5, 4
    sigma2 = 0.01
    reps = 10

    # Sample parameters
    np.random.seed(1)
    rho0, beta0 = sample_params(l, a)
    print(rho0, beta0)
    exit()
    model0 = get_model(l, a, rho0, beta0)
    params0 = {k: v.detach().numpy() for k, v in model0.kernel.get_params().items()}
    
    results, param_results = [], []
    for _ in range(reps):
        # Generate data
        X, y, y_var = simulate_data(model0, sigma2=sigma2)   
        
        for i in range(1, 10):
            p = i/10.
            train_x, train_y, test_x, test_y, train_y_var = split_data(X, y, y_var, p=p)
            
            # Predict with the true kernel
            ypred = predict(model0, train_x, train_y, train_y_var, test_x)
            record = evaluate_predictions(p, ypred, test_y)
            record['label'] = 'True kernel'
            mse1 = record['mse']
            results.append(record)
            print(record)
 
            # Train model hyperparameters   
            model = get_model(l, a, rho0, beta0)        
            params = train(model, train_x, train_y, train_y_var)
            ypred = predict(model, train_x, train_y, train_y_var, test_x)
            record = evaluate_predictions(p, ypred, test_y)
            record['label'] = 'Inferred kernel'
            mse2 = record['mse']
            results.append(record)
            print(record)
            
            # Evaluate model hyperparameters
            record = evaluate_params(params0, params, p)
            record['mse_log_ratio'] = np.log2(mse2/mse1)
            param_results.append(record)
            print(record)
    
    results = pd.DataFrame(results)
    results.to_csv('predictions.noise.csv')
    
    param_results = pd.DataFrame(param_results)
    param_results.to_csv('parameters.noise.csv')
