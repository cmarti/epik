#!/usr/bin/env python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot(axes, results, title=''):
    axes.scatter(results['p'], results['rho'], alpha=0.4, c='black')
    axes.plot((-1, 2), (0.1, 0.1), linestyle='--', c='purple', lw=0.5)
    axes.set(xlabel='Proportion of training data',
             ylabel=r'Inferred $\rho$', title=title,
             ylim=(0.02, 0.12), xlim=(-0.1, 1.1))
    axes.grid(alpha=0.5)


if __name__ == '__main__':
    ls = [5, 10]
    alphas = [4, 2]
    rho0 = 0.1
    
    fig, subplots = plt.subplots(2, 2, figsize=(4 * 2, 2 * 3.5),
                                 sharey=False, sharex=False)
    
    history = pd.read_csv('rho.inference.l5.history.csv')
    print(history)
    axes = subplots[0][0]
    sns.lineplot(x='i', y='loss', hue='p', data=history, ax=axes)
    axes.set(xlabel='Training iteration', xlim=(0, 100),
             ylabel='Loss function')
    axes.legend(loc=1)
     
    axes = subplots[0][1]
    sns.lineplot(x='i', y='rho', hue='p', data=history, ax=axes)
    axes.plot((-2, 101), (0.1, 0.1), linestyle='--', c='purple', lw=0.5)
    axes.set(xlabel='Training iteration', xlim=(0, 100),  
             ylabel=r'Kernel $\rho$', ylim=(0.02, 0.12))
    axes.legend(loc=1)
    
    axes = subplots[1][0]
    data = history.loc[history['p'] == 1, :]
    for k, df in data.groupby(['k']):
        axes.plot(df['rho'], df['loss'])
#     sns.lineplot(x='rho', y='loss', data=data, ax=axes)
    axes.set(xlabel=r'$\rho$', ylabel='Loss function')
    axes.grid(alpha=0.5)
    
    
    l = 5
    axes = subplots[1][1]
    results = pd.read_csv('rho.inference.l{}.csv'.format(l), index_col=0)
    plot(axes, results)

    fig.tight_layout()
    fig.savefig('rho.inference.png', dpi=300)