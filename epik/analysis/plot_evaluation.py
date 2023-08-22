#!/usr/bin/env python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    results = pd.read_csv('predictions.noise.csv', index_col=0)
    param_results = pd.read_csv('parameters.noise.csv', index_col=0)
    
    # Predictions
    fig, subplots = plt.subplots(2, 2, figsize=(8, 8))

    axes = subplots[0, 0]
    sns.lineplot(x='p', y='r2', hue='label',
                 data=results, ax=axes, err_style="bars")
    axes.set(xlabel='Proportion of training data',
             ylabel=r'Test $R^2$', )
    axes.legend(loc=4)
    
    axes = subplots[0, 1]
    sns.lineplot(x='p', y='mse_log_ratio',
                 data=param_results, ax=axes, err_style="bars")
    xlim = axes.get_xlim()
    axes.plot(xlim, (0, 0), lw=0.5, c='grey', linestyle='--')
    axes.set(xlabel='Proportion of training data', xlim=xlim,
             ylabel=r'Test $\log_2(MSE_{trained}/MSE_{True})$', )
    
    # parameters
    axes = subplots[1, 0]
    sns.lineplot(x='p', y='rho_r',  data=param_results, ax=axes, err_style="bars")
    axes.set(xlabel='Proportion of training data',
             ylabel=r'Pearson correlation $\log(\rho_p)$', )
    
    axes = subplots[1, 1]
    sns.lineplot(x='p', y='beta_r', data=param_results, ax=axes, err_style="bars")
    axes.set(xlabel='Proportion of training data',
             ylabel=r'Pearson correlation $\log(\eta_{p,a})$', )

    fig.tight_layout()
    fig.savefig('preditions.noise.png', dpi=300)