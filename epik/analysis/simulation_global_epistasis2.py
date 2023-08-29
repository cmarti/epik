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
from epik.src.kernel import GeneralizedSiteProductKernel, VCKernel
from epik.src.priors import AllelesProbPrior, RhosPrior
from build.lib.epik.src.priors import LambdasFlatPrior
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.kernels.rbf_kernel import RBFKernel



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
        phi = self.X_to_phi(X).flatten()
        y_true = self.phi_to_y(phi)
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
    def __init__(self, l=16, dg=1, s=1, rho_on=6, rho_off=1):
        self.l = l
        self.alleles = ['A', 'B']
        self.alpha = 2
        
        self.dg = dg
        self.s = s
        self.rho_on = rho_on
        self.rho_off = rho_off
        
        self.seq0 = 'A' * self.l
        self.X0 = seq_to_one_hot([self.seq0], alleles=self.alleles)
        self.dtype = torch.float32
        self.phi_range = (0, self.l+1)
    
    def phi_to_y(self, phi):
        f = self.s / (1 + np.exp(self.dg * (phi - self.rho_on)))
        f -= self.s / (1 + np.exp(self.dg * (phi - self.rho_off))) 
        return(f)

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
    def __init__(self, l, a, train_p=False, sites_equal=True, track_progress=False,
                 kernel='rho', rho0=None, dtype=torch.float32):
        self.l = l
        self.a = a
        if kernel == 'rho':
            rhos_prior = RhosPrior(seq_length=l, n_alleles=a, rho0=rho0,
                                   sites_equal=sites_equal, train=True, dtype=dtype)
            p_prior = AllelesProbPrior(seq_length=l, n_alleles=a, dummy_allele=False,
                                       sites_equal=True, train=train_p, dtype=dtype)
            kernel = GeneralizedSiteProductKernel(n_alleles=a, seq_length=l,
                                                  rho_prior=rhos_prior, p_prior=p_prior,
                                                  dtype=dtype)
        elif kernel == 'VC':
            lambdas_prior = LambdasFlatPrior(seq_length=l)
            kernel = VCKernel(seq_length=l, n_alleles=a, lambdas_prior=lambdas_prior,
                              dtype=dtype)
        elif kernel == 'RBF':
            kernel = RBFKernel()
        else:
            msg = 'Kernel not allowed: {}'.format(kernel)
            raise ValueError(msg)
            
        self.model = EpiK(kernel, likelihood_type='Gaussian',
                          track_progress=track_progress, dtype=dtype)
    
    def sample_params(self):
        rho0 = np.random.uniform(size=self.l)
        p0 = np.vstack([np.random.dirichlet(np.ones(self.a)) for _ in range(self.l)])
        beta0 = -np.log(p0 / (1 - p0))
        return(torch.tensor(rho0), torch.tensor(beta0))
    
    def simulate(self, X, sigma2):
        y_true = self.model.sample(X, n=1, sigma2=0).flatten()
        # y_true = (y_true - y_true.mean()) / y_true.std()
        y = torch.normal(y_true, np.sqrt(sigma2))
        y_var = sigma2 * torch.ones(X.shape[0])
        return(y_true, y, y_var)

    def predict(self, train_x, train_y, train_y_var, test_x):
        self.model.set_data(train_x, train_y, train_y_var)
        ypred = self.model.predict(test_x).detach()
        return(ypred)

    def train(self, train_x, train_y, train_y_var, n=100):
        self.model.set_data(train_x, train_y, train_y_var)
        self.model.fit(n_iter=n)
        try:
            params = self.model.kernel.get_params()
            params = {k: v.detach().numpy() for k, v in params.items()}
        except:
            return({})
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
    
    @property
    def history(self):
        return(self.model.loss_history)


def plot_predictions(axes, phi_true, y_true, xs, ys, 
                     xlabel=r'Latent phenotype $\phi$',
                     ylabel='', title='', r2=None,
                     train_y_mean=None):
    axes.scatter(phi_true, y_true, alpha=0.2, s=10, c='black',
                 label='data')
    axes.plot(xs, ys, c='grey', lw=1, label='landscape')
    axes.set(xlabel=xlabel,
             ylabel=ylabel)
    axes.grid(alpha=0.1)
    if r2 is not None:
        axes.text(0.05, 0.95, '$R^{2}$' + '={:.2f}'.format(r2),
                  transform=axes.transAxes, ha='left', va='top')
    
    if title:
        axes.text(0.05, 0.05, title,
                  transform=axes.transAxes, ha='left', va='bottom')
    
    if train_y_mean is not None:
        axes.plot((xs[0], xs[-1]), (train_y_mean, train_y_mean),
                  lw=1, linestyle='--', c='lightgrey',
                  label='empirical mean')


def plot_hist(axes, phi, bins, title=''):
    sns.histplot(phi, ax=axes, color='black', bins=bins, legend=False)
    axes.set(xlabel='', ylabel='# training sequences', title=title)
    axes.grid(alpha=0.1)


if __name__ == '__main__':
    l, alpha = 14, 2
    uniform = False
    pmut = 0.02
    n = 500
    ptrain = 0.8

    a = -3
    b = 3
    c = 3
    lower = 0
    upper = 1
    sigma2 = 0

    # Generate data
    landscape = CraterLandscape(l=l)
    X, phi, y_true, y, y_var = landscape.simulate(sigma2)
    xs, ys = landscape.get_xs_ys()
    
    
    # Split data
    ps = [0.01, 0.1, 0.5, 0.8]
    fig, subplots = plt.subplots(3, len(ps), figsize=(3.5*len(ps), 9),
                                 sharex=True, sharey='row')
    
    for p, axes in zip(ps, subplots.T):
        splits = landscape.split_data([X, y, y_var, phi], p=p, uniform=uniform, pmut=pmut)
        train_x, train_y, train_y_var, train_phi = splits[:4]
        test_x, test_y, test_y_var, test_phi = splits[4:]
        train_y_mean = train_y.mean()
        
        # Plot distance distribution
        plot_hist(axes[0], train_phi, bins=np.arange(l+1),
                  title='{:.0f}% training data'.format(p*100),)
        
        # Train fixed rho
        model = GPmodel(l, alpha, train_p=False, sites_equal=True,
                        track_progress=False, kernel='rho')
        params = model.train(train_x, train_y, train_y_var, n=50)
        test_y_pred = model.predict(train_x, train_y, train_y_var, test_x)
        evals = model.evaluate_predictions(test_y, test_y_pred)
        plot_predictions(axes[1], test_phi, test_y_pred, xs, ys,
                         xlabel='', title=r'$\rho$ kernel',
                         r2=evals['r2'], train_y_mean=train_y_mean)
        ylim = axes[1].get_ylim()
#         
#         # RBF kernel
#         model = GPmodel(l, alpha, train_p=False, sites_equal=True,
#                         track_progress=False, kernel='RBF')
#         params = model.train(train_x, train_y, train_y_var, n=50)
#         test_y_pred = model.predict(train_x, train_y, train_y_var, test_x)
#         evals = model.evaluate_predictions(test_y, test_y_pred)
#         plot_predictions(axes[2], test_phi, test_y_pred, xs, ys,
#                          xlabel='', title='RBF kernel',
#                          r2=evals['r2'], train_y_mean=train_y_mean)
        
        # VC kernel
        model = GPmodel(l, alpha, kernel='VC', track_progress=False)
        params = model.train(train_x, train_y, train_y_var, n=50)
        test_y_pred = model.predict(train_x, train_y, train_y_var, test_x)
        evals = model.evaluate_predictions(test_y, test_y_pred)
        plot_predictions(axes[2], test_phi, test_y_pred, xs, ys,
                         title=r'VC kernel', 
                         xlabel='Hamming distance from WT',
                         r2=evals['r2'], train_y_mean=train_y_mean)
        axes[2].set_ylim(ylim)
        print(train_x.shape, test_x.shape, params['lambdas'])
        
#         # Change mean
#         model = GPmodel(l, alpha, train_p=False, sites_equal=True)
#         params = model.train(train_x, train_y-c, train_y_var, n=50)
#         test_y_pred = model.predict(train_x, train_y-c, train_y_var, test_x)
#         evals = model.evaluate_predictions(test_y, test_y_pred)
#         plot_predictions(axes[2], test_phi, test_y_pred, xs, ys-c,
#                          title='', xlabel='Hamming distance from WT',
#                          r2=evals['r2'])
        
    subplots[1][0].set_ylabel('y')
    subplots[2][0].set_ylabel('y')
    
    
    fig.tight_layout()
    
    sampling = 'uniform' if uniform else 'mutation'
    fig.savefig('global_epistasis.l{}.{}.png'.format(l, sampling), dpi=300)
    
    exit()
    
    
    
    
    
    # # Train p
    # model = get_model(l, a, rho0, train_p=True)        
    # params = train(model, train_x, train_y, train_y_var)
    # ypred2 = predict(model, train_x, train_y, train_y_var, test_x)
    # record = evaluate_predictions(p, ypred2, test_y)
    # record['label'] = 'Pi kernel'
    # record.update(evaluate_params(params0, params, p))
    # print(record)
    
    
