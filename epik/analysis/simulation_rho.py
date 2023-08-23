#!/usr/bin/env python
import pandas as pd
import numpy as np
import torch

from itertools import product

from epik.src.utils import seq_to_one_hot, split_training_test
from epik.analysis.simulation_global_epistasis2 import GPmodel


if __name__ == '__main__':
    alleles = ['A', 'C', 'G', 'T']
    l, a = 5, len(alleles)
    sigma2 = 0
    reps = 5
    rho0 = torch.tensor([0.1])
    n = 100

    seqs = np.array([''.join(gt) for gt in product(alleles, repeat=l)])
    X = seq_to_one_hot(seqs, alleles)
    model = GPmodel(l, a, rho0=rho0, track_progress=False)
    y_true, y, y_var = model.simulate(X, sigma2)
    
    results = []
    history = []
    for i in [0.5] + list(range(1, 11)):
        for k in range(reps):
            p = i/10.
            splits = split_training_test(X, y, y_var, ptrain=p)
            train_x, train_y, test_x, test_y, train_y_var = splits
        
            model = GPmodel(l, a, rho0=rho0, track_progress=False)
            params = model.train(train_x, train_y, train_y_var, n=n)
            test_y_pred = model.predict(train_x, train_y, train_y_var, test_x)
            record = model.evaluate_predictions(test_y, test_y_pred)
            record.update({'k': k, 'p': p, 'rho': params['rho'][0]})
            h = pd.DataFrame({'i': np.arange(1, n+1),
                              'loss': [x['loss'] for x in model.history],
                              'rho': [x['rho'].detach().numpy()[0] for x in model.history]})
            h['k'] = k
            h['p'] = p
            history.append(h)
            results.append(record)
            print(record)
    
    results = pd.DataFrame(results)
    results.to_csv('rho.inference.l{}.csv'.format(l))
    
    history = pd.concat(history)
    history.to_csv('rho.inference.l{}.history.csv'.format(l))
