#!/usr/bin/env python
import argparse

import numpy as np
import pandas as pd
import torch

from os.path import exists
from linear_operator.settings import max_cg_iterations
from gpytorch.kernels import ScaleKernel, RQKernel, LinearKernel
from gpytorch.kernels.keops import MaternKernel

from epik.src.utils import LogTrack, guess_space_configuration, seq_to_one_hot, seq_to_binary
from epik.src.plot import plot_training_history
from epik.src.model import EpiK
from epik.src.kernel.base import AdditiveHeteroskedasticKernel
from epik.src.kernel.haploid import (VarianceComponentKernel, DeltaPKernel,
                                     RhoPiKernel, RhoKernel, AdditiveKernel,
                                     RBFKernel, ARDKernel, PairwiseKernel)


def select_kernel(kernel, n_alleles, seq_length, dtype, P, add_het, use_keops, add_scale=False,
                  binary=False):
    is_cor_kernel = True
    if kernel == 'matern':
        kernel = MaternKernel()
    elif kernel == 'RQ':
        kernel = RQKernel()
    elif kernel == 'linear':
        kernel = LinearKernel()
    # elif kernel == 'ARD':
    #     kernel = KeOpsRBF(ard_num_dims=n_alleles * seq_length)
    else:
        kernels = {'RBF': RBFKernel, 'Connectedness': RhoKernel, 'Rho': RhoKernel,
                   'ARD': ARDKernel, 'RhoPi': RhoPiKernel,
                   'Additive': AdditiveKernel, 'Pairwise': PairwiseKernel,
                   'DP': DeltaPKernel, 'VC': VarianceComponentKernel}

        is_cor_kernel = False
        if kernel not in kernels:
            msg = 'Unknown kernel provided: {}'.format(kernel)
            raise ValueError(msg)

        kwargs = {'dtype': dtype, 'use_keops': use_keops, 'binary': binary, 'P': P}
        kernel = kernels[kernel](n_alleles, seq_length, **kwargs)

    if add_het:
        kernel = AdditiveHeteroskedasticKernel(kernel, n_alleles=n_alleles,
                                               seq_length=seq_length)
        
    elif is_cor_kernel or add_scale:
        print('Correlation kernel')
        kernel = ScaleKernel(kernel)
        
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
                               help='Kernel function to use (VC, DP, Connectedness, RhoPi, RBF, ARD, Additive, RQ, matern, linear)')
#     options_group.add_argument('--q', default=None, type=float,
#                                help='Probability of leaving under the discrete time chain in sVC prior (l-1)/l')
    options_group.add_argument('-P', '--P', default=2, type=int,
                               help='P for Delta(P) prior (2)')
    options_group.add_argument('-N', '--n_kernels', default=1, type=int,
                               help='Number of kernels to learn')
    options_group.add_argument('--params', default=None,
                               help='Model parameters to use for predictions')
    options_group.add_argument('--het', default=False, action='store_true',
                               help='Add sequence dependent heteroskedasticity')
    
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
    comp_group.add_argument('--keops', default=False,
                            action='store_true', help='Use KeOps backedn')
    comp_group.add_argument('-b', '--binary', default=False, action='store_true',
                            help='Use binary encoding if number of alleles is 2')
    comp_group.add_argument('-l', '--lbfgs', default=False,
                            action='store_true', help='Use LBFGS optimizer instead of Adam')

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('-o', '--output', required=True, help='Output file')
    output_group.add_argument('-p', '--pred',
                              help='File containing sequences for predicting genotype')

    # Parse arguments
    parsed_args = parser.parse_args()
    data_fpath = parsed_args.data
    
    kernel_label = parsed_args.kernel
#     q = parsed_args.q
    P = parsed_args.P
    n_kernels = parsed_args.n_kernels
    params_fpath = parsed_args.params
    add_het = parsed_args.het
    
    gpu = parsed_args.gpu
    n_devices = parsed_args.n_devices
    n_iter = parsed_args.n_iter
    use_keops = parsed_args.keops
    learning_rate = parsed_args.learning_rate
    use_float64 = parsed_args.use_float64
    binary = parsed_args.binary
    optimizer = 'lbfgs' if parsed_args.lbfgs else 'Adam'

    pred_fpath = parsed_args.pred
    out_fpath = parsed_args.output
    
    # Load counts data
    log = LogTrack()
    log.write('Start analysis')
    data = pd.read_csv(data_fpath, index_col=0).dropna()
    log.write('Loaded {} sequences from {}'.format(data.shape[0], data_fpath))
    
    # Get processed data
    seqs = data.index.values
    test_seqs = read_test_sequences(pred_fpath)
    config = guess_space_configuration(np.append(seqs, test_seqs))
    alleles = np.unique(np.hstack(config['alphabet']))
    n_alleles = alleles.shape[0]

    if n_alleles == 2 and binary:
        X = seq_to_binary(seqs, ref=alleles[0]) # May want to make this more explicit
    else:
        X = seq_to_one_hot(seqs, alleles=alleles)
        binary = False

    y = data.values[:, 0]
    if data.shape[1] > 1:
        y_var = data.values[:, 1]
        y_var[y_var < 0.0001] = 0.0001
    else:
        y_var =  None
    dtype = torch.float64 if use_float64 else torch.float32
      
    # Get kernel
    log.write('Selected {} kernel (N={})'.format(kernel_label, n_kernels))
    add_scale = n_kernels > 1
    kernel = select_kernel(kernel_label, n_alleles, config['seq_length'],
                           dtype=dtype, P=P, add_het=add_het, use_keops=use_keops,
                           add_scale=add_scale, binary=binary)
    for i in range(1, n_kernels):
        kernel += select_kernel(kernel_label, n_alleles, config['seq_length'],
                                dtype=dtype, P=P, add_het=add_het, use_keops=use_keops,
                                add_scale=add_scale, binary=binary)

    # Define device
    device = torch.device('cuda') if gpu else None
    device_label = 'GPU' if gpu else 'CPU'
    log.write('Running computations on {}'.format(device_label))
    
    # Create model
    with max_cg_iterations(3000):
        log.write('Building model for Gaussian Process regression')
        model = EpiK(kernel, dtype=dtype, track_progress=True,
                     train_noise=True,
                     device=device, n_devices=n_devices,
                     learning_rate=learning_rate, optimizer=optimizer)
        model.set_data(X, y, y_var=y_var)
    
        # Load hyperparameters if provided
        if params_fpath is not None:
            if not exists(params_fpath):
                log.write('Hyperparameters file not found: {}'.format(params_fpath))
            else:
                log.write('Load hyperparameters from {}'.format(params_fpath))
                model.load(params_fpath)
        
        # Fit by evidence maximization
        log.write('Train hyperparameters by maximizing the evidence')
        model.fit(n_iter=n_iter)
    
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
            
    log.finish()
    
    
if __name__ == '__main__':
    main()
