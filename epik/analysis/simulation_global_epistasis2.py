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

from epik.src.utils import seq_to_one_hot, get_tensor
from epik.src.model import EpiK
from epik.src.kernel import GeneralizedSiteProductKernel
from epik.src.priors import AllelesProbPrior, RhosPrior



class GlobalEpistasisModel(object):
    @property
    def n(self):
        return(self.alpha ** self.l)
    
    def add_noise(self, y_true, sigma2=0):
        if sigma2 == 0:    
            y = torch.normal(y_true, np.sqrt(sigma2))
            y_var = sigma2 * torch.ones(y_true.shape[0])
        else:
            y = y_true.copy()
            y_var = torch.zeros_like(y)
        return(y, y_var)
    
    def calc_hamming_distance(self, x1, x2):
        return(self.l - x1.matmul(self.X2.T))

    def get_sequences(self):
        seqs = np.array([''.join(gt) for gt in product(self.alleles, repeat=self.l)])
        return(seqs)
    
    def seqs_to_X(self, seqs):
        return(seq_to_one_hot(seqs, self.alleles))
    
    def simulate(self, sigma2=0):
        seqs = self.get_sequences()
        X = self.seqs_to_X(seqs)
        phi = self.X_to_phi(X)
        y_true = self.phi_to_y(phi).flatten()
        y, y_var = self.add_noise(y_true, sigma2=sigma2)
        return(X, phi, y_true, y, y_var)
    
    def calc_sampling_ps(self, X, pmut=0.1):
        site_log_ps = [np.log(1-pmut)] + [np.log(pmut/(self.alpha-1))] * (self.alpha-1)
        site_log_ps = get_tensor(site_log_ps * self.l, dtype=self.dtype).flatten()
        ps = np.exp(X.matmul(site_log_ps))
        return(ps)
    
    def get_train_test_split(self, X, n, uniform=True, pmut=0.1):
        
        if not uniform:
            p = self.calc_sampling_ps(X, pmut).numpy()
        else:
            p = 1 / X.shape[0] * np.ones(X.shape[0])
        
        idx = np.arange(X.shape[0])
        idx = np.random.choice(idx, p=p, size=n, replace=False)
        train = np.full(X.shape[0], fill_value=False)
        train[idx] = True
        test = ~train
        return(train, test)
            
    def split_data(self, data, p=0.8, uniform=True, pmut=0.1):
        X = data[0]
        n = int(X.shape[0] * p)
        train, test = self.get_train_test_split(X, n, uniform=uniform, pmut=pmut)
        output = [get_tensor(a[train], dtype=self.dtype) for a in data]
        output.extend([get_tensor(a[test], dtype=self.dtype) for a in data])
        return(output)
    
    def get_xs_ys(self):
        xs = np.linspace(self.phi_range[0], self.phi_range[1], 101)
        ys = self.phi_to_y(xs)
        return(xs, ys)


class CraterLandscape(GlobalEpistasisModel):
    def __init__(self, l=16, d0=3, lengthscale=1):
        self.l = l
        self.alleles = ['A', 'B']
        self.alpha = 2
        
        self.d0 = d0
        self.lengthscale = lengthscale
        self.seq0 = 'A' * self.l
        self.X0 = seq_to_one_hot([self.seq0], alleles=self.alleles)
        self.dtype = torch.float32
        self.phi_range = (0, self.l+1)
    
    def phi_to_y(self, phi):
        return(np.exp(-(phi - self.d0)**2 / self.lengthscale))

    def X_to_phi(self, X):
        return(self.l - X.matmul(self.X0.T))


class SigmoidLandscape(GlobalEpistasisModel):
    def __init__(self, l, alleles=['A', 'C', 'G', 'T'],
                 a=0, b=1, upper=1, lower=0):
        self.l = l
        self.alleles = alleles
        self.alpha = len(alleles)
        
        self.a = a
        self.b = b
        self.upper = upper
        self.lower = lower
        
    def X_to_phi(self, X):
        print(self.seq0, self.X0.shape)
        return(self.l - X.matmul(self.X0.T))    
    
    def get_thetas(self, fixed=False):
        a = self.alpha
        l = self.l
        if fixed:
            theta = np.vstack([[[(a-1) / a] + [-1/a] * (a-1)] * l])
        else:
            theta = np.random.normal(size=(l, a))
        theta = theta - np.expand_dims(theta.mean(1), axis=1)
        return(torch.tensor(theta.flatten(), dtype=torch.float32))
    
    def phi_to_y(self, phi):
        y_raw = np.exp(self.a + self.b * phi)
        return(self.lower + (self.upper - self.lower) * y_raw / (1 + y_raw))


class GPmodel(object):
    def __init__(self, l, a, train_p=False, sites_equal=True, track_progress=False):
        self.l = l
        self.a = a
        rhos_prior = RhosPrior(seq_length=l, n_alleles=a,
                               sites_equal=sites_equal, train=True)
        p_prior = AllelesProbPrior(seq_length=l, n_alleles=a, dummy_allele=False,
                                   sites_equal=False, train=train_p)
        
        kernel = GeneralizedSiteProductKernel(n_alleles=a, seq_length=l,
                                              rho_prior=rhos_prior, p_prior=p_prior)
        self.model = EpiK(kernel, likelihood_type='Gaussian',
                          track_progress=track_progress)
    
    def sample_params(self):
        rho0 = np.random.uniform(size=self.l)
        p0 = np.vstack([np.random.dirichlet(np.ones(self.a)) for _ in range(self.l)])
        beta0 = -np.log(p0 / (1 - p0))
        return(torch.tensor(rho0), torch.tensor(beta0))

    def predict(self, train_x, train_y, train_y_var, test_x):
        self.model.set_data(train_x, train_y, train_y_var)
        ypred = self.model.predict(test_x).detach()
        return(ypred)

    def train(self, train_x, train_y, train_y_var, n=100):
        self.model.set_data(train_x, train_y, train_y_var)
        self.model.fit(n_iter=n)
        params = self.model.kernel.get_params()
        params = {k: v.detach().numpy() for k, v in params.items()}
        return(params)

    def evaluate_predictions(self, test_y, test_y_pred):
        ypred, test_y = test_y_pred.numpy(), test_y.numpy()
        r2 = pearsonr(ypred, test_y)[0] ** 2
        mse = np.mean((ypred - test_y) ** 2)
        return({'r2': r2, 'mse': mse})

    def evaluate_params(self, params0, params):
        logrho0 = np.log(params0['rho'])
        logrho = np.log(params['rho'])
        rho_r = pearsonr(logrho, logrho0)[0]
        beta0 = params0['beta'].flatten()
        beta = params['beta'].flatten()
        beta_r = pearsonr(beta, beta0)[0]
        return({'rho_r': rho_r, 'beta_r': beta_r})


def plot_predictions(axes, phi_true, y_true, xs, ys, 
                     xlabel=r'Latent phenotype $\phi$',
                     ylabel='', title='', xlims=(0, 14)):
    axes.scatter(phi_true, y_true, alpha=0.2, s=10, c='black')
    axes.plot(xs, ys, c='grey', lw=1)
    axes.set(xlabel=xlabel,
             ylabel=ylabel,
             xlim=xlims, title=title)
    axes.grid(alpha=0.1)


def plot_hist(axes, phi, bins, title=''):
    sns.histplot(phi, ax=axes, color='black', bins=bins, legend=False)
    axes.set(xlabel='', ylabel='# training sequences', title=title)
    axes.grid(alpha=0.1)


if __name__ == '__main__':
    l, alpha = 11, 2
    pmut = 0.1
    n = 500
    ptrain = 0.8

    lengthscale = 3
    a = -3
    b = 3
    lower = 0
    upper = 1
    sigma2 = 0

    # Generate data
    landscape = CraterLandscape(l=l, d0=3, lengthscale=lengthscale)
    X, phi, y_true, y, y_var = landscape.simulate(sigma2)
    xs, ys = landscape.get_xs_ys()
    
    
    # Split data
    ps = [0.01, 0.1, 0.5, 0.8]
    fig, subplots = plt.subplots(3, len(ps), figsize=(4*len(ps), 9),
                                 sharex=True, sharey=False)
    
    for p, axes in zip(ps, subplots.T):
        splits = landscape.split_data([X, y, y_var, phi], p=p, uniform=False, pmut=pmut)
        train_x, train_y, train_y_var, train_phi = splits[:4]
        test_x, test_y, test_y_var, test_phi = splits[4:]
        
        # Plot distance distribution
        plot_hist(axes[0], train_phi, bins=np.arange(l+1),
                  title=r'Uniform $\rho$' + ' ({:.0f}% training)'.format(p*100),)
        
        # Train fixed rho
        model = GPmodel(l, alpha, train_p=False, sites_equal=True,
                        track_progress=False)
        params = model.train(train_x, train_y, train_y_var, n=50)
        test_y_pred = model.predict(train_x, train_y, train_y_var, test_x)
        
        print(train_x.shape, test_x.shape, params['rho'][0], model.evaluate_predictions(test_y, test_y_pred))
        
        plot_predictions(axes[1], test_phi, test_y_pred, xs, ys,
                         xlabel='', ylabel='y', title='')
        
        # # Change mean
        # model = GPmodel(l, alpha, train_p=False, sites_equal=True)
        # params = model.train(train_x, train_y-1, train_y_var, n=100)
        # test_y_pred = model.predict(train_x, train_y-1, train_y_var, test_x)
        # plot_predictions(axes[2], test_phi, test_y_pred, xs, ys-1,
        #                  title='', xlabel='Hamming distance from WT',
        #                  ylabel='y')
        
        
    
    #
    # record = evaluate_predictions(ptrain, test_y_pred, test_y)
    # record['rho'] = params['rho']
    # print(record)
    # print(params0['rho'])
    #
    # axes = subplots[1]
    # plot_true_predictions(axes, phi_train, train_y_pred, a=a, b=b,
    #                       lower=0, upper=1)
    #
    # axes = subplots[2]
    # plot_true_predictions(axes, phi_test, test_y_pred, a=a, b=b,
    #                       lower=0, upper=1)
    
    fig.tight_layout()
    fig.savefig('global_epistasis.png', dpi=300)
    
    exit()
    
    
    
    
    
    # # Train p
    # model = get_model(l, a, rho0, train_p=True)        
    # params = train(model, train_x, train_y, train_y_var)
    # ypred2 = predict(model, train_x, train_y, train_y_var, test_x)
    # record = evaluate_predictions(p, ypred2, test_y)
    # record['label'] = 'Pi kernel'
    # record.update(evaluate_params(params0, params, p))
    # print(record)
    
    
