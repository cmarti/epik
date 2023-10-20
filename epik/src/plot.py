import matplotlib.pyplot as plt


def plot_training_history(loss_history, fpath):
    fig, axes = plt.subplots(1, 1, figsize=(4, 3))
    
    axes.plot(-loss_history)
    axes.set(xlabel='Iteration', ylabel='Marginal log-Likelihood',
             title='Hyperparameter optimization')
    
    fig.tight_layout()
    fig.savefig(fpath)
    