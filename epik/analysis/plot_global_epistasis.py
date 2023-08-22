#!/usr/bin/env python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    results = pd.read_csv('predictions.global_epistasis.csv', index_col=0)
    results['rho_r2'] = results['rho_r'] ** 2
    
    # Predictions
    fig, subplots = plt.subplots(1, 2, figsize=(8, 4))

    axes = subplots[0]
    sns.lineplot(x='p', y='r2', hue='label',
                 data=results, ax=axes, err_style="bars")
    axes.set(xlabel='Proportion of training data',
             ylabel=r'Test $R^2$', ylim=(0, 1))
    axes.legend(loc=4)
    axes.grid(alpha=0.1)
    
    # parameters
    axes = subplots[1]
    sns.lineplot(x='p', y='rho_r2', hue='label',
                 data=results, ax=axes, err_style="bars")
    axes.set(xlabel='Proportion of training data',
             ylabel=r'$\log(\rho_p)\ R^2$', ylim=(0, 1))
    axes.legend(loc=4)
    axes.grid(alpha=0.1)
    
    fig.tight_layout()
    fig.savefig('preditions.global_epistasis.png', dpi=300)