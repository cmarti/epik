#!/usr/bin/env python
import argparse

import numpy as np
import pandas as pd
import torch

from gpytorch.settings import max_cg_iterations
from gpytorch.kernels import ScaleKernel, RQKernel, LinearKernel
from gpytorch.kernels.keops import RBFKernel, MaternKernel

from epik.src.kernel import (VarianceComponentKernel, DeltaPKernel,
                             HetRBFKernel)
from epik.src.keops import (RhoPiKernel, RhoKernel)

from epik.src.model import EpiK
from epik.src.utils import LogTrack, guess_space_configuration, seq_to_one_hot
from epik.src.plot import plot_training_history


def select_kernel(kernel, n_alleles, seq_length, dtype, P):
    if kernel == 'RBF':
        kernel = ScaleKernel(RBFKernel())
    elif kernel == 'ARD':
        kernel = ScaleKernel(RBFKernel(ard_num_dims=n_alleles * seq_length))
    elif kernel == 'matern':
        kernel = ScaleKernel(MaternKernel())
    elif kernel == 'RQ':
        kernel = ScaleKernel(RQKernel())
    elif kernel == 'linear':
        kernel = ScaleKernel(LinearKernel())
    else:
        if kernel == 'Connectedness' or kernel == 'Rho':
            kernel = RhoKernel(n_alleles, seq_length, dtype=dtype)
        elif kernel == 'RhoPi':
            kernel = RhoPiKernel(n_alleles, seq_length, dtype=dtype)
        elif kernel == 'HetRBF':
            kernel = HetRBFKernel(n_alleles, seq_length, dtype=dtype)
        elif kernel == 'HetARD':
            kernel = HetRBFKernel(n_alleles, seq_length, dtype=dtype,
                                  dims=n_alleles * seq_length)
        elif kernel == 'VC':
            kernel = VarianceComponentKernel(n_alleles, seq_length, dtype=dtype)
        elif kernel == 'DP':
            kernel = DeltaPKernel(n_alleles, seq_length, P=P, dtype=dtype)
        else:
            msg = 'Unknown kernel provided: {}'.format(kernel)
            raise ValueError(msg)
    return(kernel)


def read_test_sequences(test_fpath):
    seqs = np.array([])
    if test_fpath is not None:
        seqs = np.array([line.strip().strip('"') for line in open(test_fpath)])
    return(seqs)
        
        
def main():
    description = 'Runs Gaussian-Process regression on sequence space using data'
    description += ' from quantitative phenotypes associated to their corresponding'
    description += ' sequences. If provided, the variance of the estimated '
    description += ' quantitative measure can be incorporated into the model'
    
    # Create arguments
    parser = argparse.ArgumentParser(description=description)
    input_group = parser.add_argument_group('Input')
    input_group.add_argument('data', help='CSV table with genotype-phenotype data')

    options_group = parser.add_argument_group('Kernel options')
    options_group.add_argument('-k', '--kernel', default='VC',
                               help='Kernel function to use (VC, DP, Connectedness, RhoPi, RBF, ARD, RQ, matern, linear)')
#     options_group.add_argument('--q', default=None, type=float,
#                                help='Probability of leaving under the discrete time chain in sVC prior (l-1)/l')
    options_group.add_argument('-P', '--P', default=2, type=int,
                               help='P for Delta(P) prior (2)')
    options_group.add_argument('--params', default=None,
                               help='Model parameters to use for predictions')
    
    comp_group = parser.add_argument_group('Computational options')
    comp_group.add_argument('--gpu', default=False, action='store_true',
                            help='Use GPU-acceleration')
    comp_group.add_argument('--use_float64', default=False, action='store_true',
                            help='Use float64 data type')
    comp_group.add_argument('-m', '--n_devices', default=1, type=int,
                            help='Number of GPUs to use (1)')
    comp_group.add_argument('-n', '--n_iter', default=200, type=int,
                            help='Number of iterations for optimization (200)')
    comp_group.add_argument('-r', '--learning_rate', default=0.1, type=float,
                            help='Learning rate for optimization (0.1)')
    comp_group.add_argument('-l', '--lbfgs', default=False,
                            action='store_true', help='Use LBFGS optimizer instead of Adam')

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('-o', '--output', required=True, help='Output file')
    output_group.add_argument('-p', '--pred',
                              help='File containing sequences for predicting genotype')

    # Parse arguments
    parsed_args = parser.parse_args()
    data_fpath = parsed_args.data
    
    kernel = parsed_args.kernel
#     q = parsed_args.q
    P = parsed_args.P
    params_fpath = parsed_args.params
    
    gpu = parsed_args.gpu
    n_devices = parsed_args.n_devices
    n_iter = parsed_args.n_iter
    learning_rate = parsed_args.learning_rate
    use_float64 = parsed_args.use_float64
    optimizer = 'lbfgs' if parsed_args.lbfgs else 'Adam'

    pred_fpath = parsed_args.pred
    out_fpath = parsed_args.output
    
    # Load counts data
    log = LogTrack()
    log.write('Start analysis')
    data = pd.read_csv(data_fpath, index_col=0).dropna()
    
    # Get processed data
    seqs = data.index.values
    test_seqs = read_test_sequences(pred_fpath)
    config = guess_space_configuration(np.append(seqs, test_seqs))
    alleles = np.unique(np.hstack(config['alphabet']))
    n_alleles = alleles.shape[0]
    X = seq_to_one_hot(seqs, alleles=alleles)
    y = data.values[:, 0]
    if data.shape[1] > 1:
        y_var = data.values[:, 1]
        y_var[y_var < 0.0001] = 0.0001
    else:
        y_var =  None
    dtype = torch.float64 if use_float64 else torch.float32
      
    # Get kernel
    log.write('Selected {} kernel'.format(kernel))
    kernel = select_kernel(kernel, n_alleles, config['seq_length'],
                           dtype=dtype, P=P)

    # Define device
    device = torch.device('cuda') if gpu else None
    device_label = 'GPU' if gpu else 'CPU'
    log.write('Running computations on {}'.format(device_label))
    
    # Create model
    max_cg_iterations(3000)
    log.write('Building model for Gaussian Process regression')
    model = EpiK(kernel, dtype=dtype, track_progress=True,
                 device=device, n_devices=n_devices,
                 learning_rate=learning_rate, optimizer=optimizer)
    model.set_data(X, y, y_var=y_var)

    # Fit by evidence maximization
    if params_fpath is None:
        log.write('Estimate hyperparameters by maximizing the evidence')
        model.fit(n_iter=n_iter)
    else:
        log.write('Load hyperparameters from {}'.format(params_fpath))
        model.load(params_fpath)

    # Write output parameters
    fpath = '{}.model_params.pth'.format(out_fpath)
    log.write('Storing model parameteres at {}'.format(fpath))
    model.save(fpath)
    
    # Predict phenotype in new sequences
    if test_seqs.shape[0] > 0:
        log.write('Obtain phenotypic predictions for test data')
        X_test = seq_to_one_hot(test_seqs, alleles=alleles)
        y_test = model.to_numpy(model.predict(X_test))
        result = pd.DataFrame({'y_pred': y_test}, index=test_seqs)
        log.write('\tWriting predictions to {}'.format(out_fpath))
        result.to_csv(out_fpath)

    # Write execution time for tracking performance
    fpath = '{}.time.txt'.format(out_fpath)
    log.write('Writing execution times to {}'.format(fpath))        
    with open(fpath, 'w') as fhand:
        fhand.write('fit,{}\n'.format(model.fit_time))
        if hasattr(model, 'pred_time'):
            fhand.write('pred,{}\n'.format(model.pred_time))
            
    # Save plot with training history
    if hasattr(model, 'loss_history'):
        fpath = '{}.training.png'.format(out_fpath)
        log.write('Saving plot with loss history to {}'.format(fpath))
        plot_training_history(np.array(model.loss_history), fpath)
    
    log.finish()


if __name__ == '__main__':
    main()
