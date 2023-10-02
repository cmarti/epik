#!/usr/bin/env python
import argparse

import numpy as np
import pandas as pd
import torch

from gpytorch.settings import max_cg_iterations
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.rq_kernel import RQKernel
from gpytorch.kernels.linear_kernel import LinearKernel

from epik.src.kernel import (SkewedVCKernel, VCKernel,
                             GeneralizedSiteProductKernel)
from epik.src.model import EpiK
from epik.src.priors import (LambdasExpDecayPrior, AllelesProbPrior,
                             LambdasFlatPrior, LambdasMonotonicDecayPrior,
                             LambdasDeltaPrior, RhosPrior)
from epik.src.utils import (LogTrack, guess_space_configuration, seq_to_one_hot,
                            get_tensor)
from epik.src.plot import plot_training_history

        
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
                               help='Kernel function to use (VC, sVC, SiteProduct, GeneralizedSiteProduct, Diploid, RBF, RQ, matern, linear)')
    options_group.add_argument('--q', default=None, type=float,
                               help='Probability of leaving under the discrete time chain in sVC prior (l-1)/l')
    options_group.add_argument('--train_p', action='store_true', default=False,
                               help='Train p parameters from the kernel')
    options_group.add_argument('--dummy_allele', action='store_true', default=False,
                               help='Add dummy allele to each site')
    options_group.add_argument('--lprior', default=None,
                               help='Type of prior on log(lambdas) {None, delta, monotonic_decay, 2nd_order_diff}')
    options_group.add_argument('-P', '--P', default=2, type=int,
                               help='P for Delta(P) prior (2)')
    
    
    comp_group = parser.add_argument_group('Computational options')
    comp_group.add_argument('--gpu', default=False, action='store_true',
                            help='Use GPU-acceleration')
    comp_group.add_argument('--use_float64', default=False, action='store_true',
                            help='Use float64 data type')
    comp_group.add_argument('-m', '--n_devices', default=1, type=int,
                            help='Number of GPUs to use (1)')
    comp_group.add_argument('-s', '--partition_size', default=0, type=int,
                            help='Use kernel partitioning on GPU of this size')
    comp_group.add_argument('-t', '--preconditioner_size', default=0, type=int,
                            help='Size of the preconditioner for CG-solve Kx=y')
    comp_group.add_argument('-n', '--n_iter', default=200, type=int,
                            help='Number of iterations for optimization (200)')
    comp_group.add_argument('-r', '--learning_rate', default=0.1, type=float,
                            help='Learning rate for optimization (0.1)')

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('-o', '--output', required=True, help='Output file')
    output_group.add_argument('-p', '--pred',
                              help='File containing sequences for predicting genotype')

    # Parse arguments
    parsed_args = parser.parse_args()
    data_fpath = parsed_args.data
    
    kernel = parsed_args.kernel
    q = parsed_args.q
    lambdas_prior = parsed_args.lprior
    P = parsed_args.P
    train_p = parsed_args.train_p
    dummy_allele = parsed_args.dummy_allele
    
    gpu = parsed_args.gpu
    n_devices = parsed_args.n_devices
    n_iter = parsed_args.n_iter
    learning_rate = parsed_args.learning_rate
    partition_size = parsed_args.partition_size
    preconditioner_size = parsed_args.preconditioner_size
    use_float64 = parsed_args.use_float64

    pred_fpath = parsed_args.pred
    out_fpath = parsed_args.output
    
    # Load counts data
    log = LogTrack()
    log.write('Start analysis')
    data = pd.read_csv(data_fpath, dtype=str)
    data = data.set_index(data.columns[0]).astype(float)
    
    # Get processed data
    seqs = data.index.values
    if pred_fpath is None:
        pred_seqs = np.array([])
    else:
        pred_seqs = np.array([line.strip().strip('"')
                              for line in open(pred_fpath)]) 
    
    config = guess_space_configuration(np.append(seqs, pred_seqs))
    alleles = np.unique(np.hstack(config['alphabet']))
    X = seq_to_one_hot(seqs, alleles=alleles)
    y = data.values[:, 0]
    if data.shape[1] > 1:
        y_var = data.values[:, 1]
        y_var[y_var < 0.0001] = 0.0001
    else:
        y_var =  None
    
    if use_float64:
        X = get_tensor(X, torch.float64)
        y = get_tensor(y, torch.float64)
        if y_var is not None:
            y_var = get_tensor(y_var, torch.float64)
        dtype = torch.float64
    else:
        dtype = torch.float32
      
    # Get kernel
    log.write('Selected {} kernel'.format(kernel))
    if kernel == 'RBF':
        kernel = ScaleKernel(RBFKernel())
    elif kernel == 'ARD':
        kernel = ScaleKernel(RBFKernel(ard_num_dims=X.shape[1]))
    elif kernel == 'matern':
        kernel = ScaleKernel(MaternKernel())
    elif kernel == 'RQ':
        kernel = ScaleKernel(RQKernel())
    elif kernel == 'linear':
        kernel = ScaleKernel(LinearKernel())
    else:
        n_alleles, seq_length = np.max(config['n_alleles']), config['length']
        p_prior = AllelesProbPrior(seq_length=seq_length, n_alleles=n_alleles,
                                   train=train_p, dtype=dtype, dummy_allele=dummy_allele)
        
        log.write('Use {} prior on lambdas'.format(lambdas_prior))
        if lambdas_prior is None:
            lambdas_prior = LambdasFlatPrior(seq_length=seq_length, dtype=dtype)
        elif lambdas_prior == 'delta':
            lambdas_prior = LambdasDeltaPrior(seq_length, n_alleles, P=P, dtype=dtype)
        elif lambdas_prior == '2nd_order_diff':
            lambdas_prior = LambdasExpDecayPrior(seq_length=seq_length, dtype=dtype)
        elif lambdas_prior == 'monotonic_decay':
            lambdas_prior = LambdasMonotonicDecayPrior(seq_length=seq_length, dtype=dtype)
        else:
            msg = 'Lambdas prior unknown: {}'.format(lambdas_prior)
            raise ValueError(msg)    
        
        if kernel == 'GeneralizedSiteProduct':
            rho_prior = RhosPrior(seq_length=seq_length, n_alleles=n_alleles)
            kernel = GeneralizedSiteProductKernel(n_alleles=n_alleles, seq_length=seq_length,
                                                  p_prior=p_prior, rho_prior=rho_prior,
                                                  dtype=dtype)
        elif kernel == 'VC':
            kernel = VCKernel(n_alleles=n_alleles, seq_length=seq_length,
                              lambdas_prior=lambdas_prior, dtype=dtype)
        elif kernel == 'sVC':
            kernel = SkewedVCKernel(n_alleles=n_alleles, seq_length=seq_length, 
                                    lambdas_prior=lambdas_prior, p_prior=p_prior,
                                    q=q, dtype=dtype)
        elif kernel == 'Diploid':
            msg = 'Diploid kernel Not implemented yet'
            raise ValueError(msg)
        else:
            msg = 'Unknown kernel provided: {}'.format(kernel)
            raise ValueError(msg)
    
    # Create model
    max_cg_iterations(3000)
    log.write('Building model for Gaussian Process regression')
    output_device = torch.device('cuda') if gpu else None
    model = EpiK(kernel, likelihood_type='Gaussian', dtype=dtype,
                 output_device=output_device, n_devices=n_devices,
                 partition_size=partition_size, learning_rate=learning_rate,
                 preconditioner_size=preconditioner_size)
    model.set_data(X, y, y_var=y_var)

    # Fit by evidence maximization
    log.write('Estimate variance components by maximizing the evidence')
    model.fit(n_iter=n_iter)

    # Output file prefix    
    prefix = '.'.join(out_fpath.split('.')[:-1])
    
    # Write output parameters
    if hasattr(kernel, 'p'):
        fpath = '{}.p.csv'.format(prefix)
        log.write('Writing inferred p to {}'.format(fpath))
        ps = pd.DataFrame(model.to_numpy(kernel.p), columns=np.append(alleles, '*'))
        ps.to_csv(fpath)
    
    if hasattr(kernel, 'lambdas'):
        fpath = '{}.lambdas.txt'.format(prefix)
        log.write('Writing inferred lambdas to {}'.format(fpath))
        with open(fpath, 'w') as fhand:
            for l in model.to_numpy(kernel.lambdas):
                fhand.write('{}\n'.format(l))
    
    if hasattr(kernel, 'theta'):
        fpath = '{}.theta.txt'.format(prefix)
        log.write('Writing inferred theta to {}'.format(fpath))
        theta = pd.DataFrame(model.to_numpy(kernel.theta))
        theta.to_csv(fpath)
    
    if hasattr(kernel, 'rho'):
        fpath = '{}.rho.txt'.format(prefix)
        log.write('Writing inferred rho to {}'.format(fpath))
        rho = pd.DataFrame(model.to_numpy(kernel.rho))
        rho.to_csv(fpath)
    
    if hasattr(kernel, 'beta'):
        fpath = '{}.beta.txt'.format(prefix)
        log.write('Writing inferred beta to {}'.format(fpath))
        if dummy_allele:
            alleles = np.append(alleles, '*')
            
        beta = pd.DataFrame(model.to_numpy(kernel.beta), columns=alleles)
        beta.to_csv(fpath)
    
    # Predict phenotype in new sequences 
    if pred_seqs.shape[0] > 0:
        log.write('Obtain phenotypic predictions')
        Xpred = seq_to_one_hot(pred_seqs, alleles=alleles)
        result = pd.DataFrame({'ypred': model.to_numpy(model.predict(Xpred))},
                              index=pred_seqs)
        log.write('\tWriting predictions to {}'.format(out_fpath))
        result.to_csv(out_fpath)

    # Write execution time for tracking performance
    fpath = '{}.time.txt'.format(prefix)
    log.write('Writing execution times to {}'.format(fpath))        
    with open(fpath, 'w') as fhand:
        fhand.write('fit,{}\n'.format(model.fit_time))
        if hasattr(model, 'pred_time'):
            fhand.write('pred,{}\n'.format(model.pred_time))
            
    # Save plot with training history
    if hasattr(model, 'loss_history'):
        fpath = '{}.training.png'.format(prefix)
        log.write('Saving plot with loss history to {}'.format(fpath))
        plot_training_history(model.loss_history, fpath)
    
    log.finish()


if __name__ == '__main__':
    main()
