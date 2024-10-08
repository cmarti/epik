#!/usr/bin/env python
import argparse

import numpy as np
import pandas as pd
import torch

from os.path import exists
from linear_operator.settings import max_cg_iterations

from epik.src.utils import LogTrack, guess_space_configuration, encode_seqs
from epik.src.model import EpiK
from epik.src.kernel import get_kernel
from epik.src.settings import KERNELS


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
    options_group.add_argument('-k', '--kernel', required=True,
                               help='Kernel function to use {}'.format(KERNELS))
    help_msg = 'pth file with model parameters'
    options_group.add_argument('--params', default=None, help=help_msg)
    
    comp_group = parser.add_argument_group('Computational options')
    comp_group.add_argument('--gpu', default=False, action='store_true',
                            help='Use GPU for computation')
    comp_group.add_argument('-m', '--n_devices', default=1, type=int,
                            help='Number of GPUs to use (1)')
    comp_group.add_argument('-n', '--n_iter', default=200, type=int,
                            help='Number of iterations for optimization (200)')
    comp_group.add_argument('-r', '--learning_rate', default=0.1, type=float,
                            help='Learning rate for optimization (0.1)')

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('-o', '--output', required=True, help='Output file')
    output_group.add_argument('-p', '--pred',
                              help='File containing sequences for predicting genotype')
    help_msg = 'Sequence context in which to predict mutational effects'
    help_msg += ' and epistatic coefficients'
    output_group.add_argument('-s', '--seq0', default=None, help=help_msg)
    output_group.add_argument('--calc_variance', default=False, action='store_true',
                               help='Compute posterior variances')

    # Parse arguments
    parsed_args = parser.parse_args()
    data_fpath = parsed_args.data
    
    kernel_label = parsed_args.kernel
    params_fpath = parsed_args.params
    
    gpu = parsed_args.gpu
    n_devices = parsed_args.n_devices
    n_iter = parsed_args.n_iter
    learning_rate = parsed_args.learning_rate

    pred_fpath = parsed_args.pred
    out_fpath = parsed_args.output
    seq0 = parsed_args.seq0
    calc_variance = parsed_args.calc_variance
    
    # Initialize logger
    log = LogTrack()
    log.write('Start analysis')

    # Load data
    data = pd.read_csv(data_fpath, index_col=0).dropna()
    log.write('Loaded {} sequences from {}'.format(data.shape[0], data_fpath))
    
    # Get processed data
    seqs = data.index.values
    test_seqs = read_test_sequences(pred_fpath)
    config = guess_space_configuration(np.append(seqs, test_seqs))
    alleles = np.unique(np.hstack(config['alphabet']))
    n_alleles = alleles.shape[0]

    X = encode_seqs(seqs, alphabet=alleles)
    y = data.values[:, 0]

    if data.shape[1] > 1:
        y_var = data.values[:, 1]
        y_var[y_var < 0.0001] = 0.0001
    else:
        y_var =  None
      
    # Get kernel
    log.write('Selected {} kernel'.format(kernel_label))
    kernel = get_kernel(kernel_label, n_alleles, config['seq_length'],
                        add_scale=kernel_label == 'GeneralProduct',
                        random_init=True)
    
    # Define device
    device = torch.device('cuda') if gpu else None
    device_label = 'GPU' if gpu else 'CPU'
    log.write('Running computations on {}'.format(device_label))
    
    # Create model
    with max_cg_iterations(1000):
        log.write('Building model for Gaussian Process regression')
        model = EpiK(kernel, track_progress=True, train_noise=True,
                     device=device, n_devices=n_devices,
                     learning_rate=learning_rate)
        model.set_data(X, y, y_var=y_var)
    
        # Load hyperparameters if provided
        if params_fpath is not None:
            if not exists(params_fpath):
                log.write('Hyperparameters file not found: {}'.format(params_fpath))
            else:
                log.write('Load hyperparameters from {}'.format(params_fpath))
                model.load(params_fpath)
        
        # Fit by evidence maximization
        if n_iter > 0:
            log.write('Train hyperparameters by maximizing the evidence')
            model.fit(n_iter=n_iter)
        
        fpath = '{}.model_params.pth'.format(out_fpath)
        log.write('Storing model parameteres at {}'.format(fpath))
        model.save(fpath)
        
        # Predict phenotype in new sequences
        if test_seqs.shape[0] > 0:
            log.write('Obtain phenotypic predictions for test data')
            X_test = encode_seqs(test_seqs, alphabet=alleles)
            result = model.predict(X_test, calc_variance=calc_variance,
                                   labels=test_seqs)
            log.write('\tWriting predictions to {}'.format(out_fpath))
            result.to_csv(out_fpath)
        
        if seq0 is not None:
            log.write('Estimating mutational effects and epistatic coefficients around {}'.format(seq0))
            df1 = model.predict_mut_effects(seq0, alleles, calc_variance=True)
            df2 = model.predict_epistatic_coeffs(seq0, alleles, calc_variance=True)
            results = pd.concat([df1, df2])
            fpath = '{}.{}_expansion.csv'.format(out_fpath, seq0)
            log.write('\tWriting estimates to {}'.format(fpath))
            results.to_csv(fpath)
    
    # Write execution time for tracking performance
    fpath = '{}.time.txt'.format(out_fpath)
    log.write('Writing execution times to {}'.format(fpath))        
    with open(fpath, 'w') as fhand:
        fhand.write('fit,{}\n'.format(model.fit_time))
        if hasattr(model, 'pred_time'):
            fhand.write('pred,{}\n'.format(model.pred_time))
        if hasattr(model, 'contrast_time'):
            fhand.write('contrast,{}\n'.format(model.contrast_time))
            
    log.finish()
    
    
if __name__ == '__main__':
    main()
