#!/usr/bin/env python
import argparse

import numpy as np
import pandas as pd
import torch

from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.rq_kernel import RQKernel
from gpytorch.kernels.linear_kernel import LinearKernel

from epik.src.kernel import SkewedVCKernel, VCKernel
from epik.src.model import EpiK
from epik.src.utils import LogTrack, guess_space_configuration, seq_to_one_hot

        
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
                               help='Kernel function to use (VC, Diploid, RBF, RQ, matern, linear)')
    help_msg = 'Standard deviation of deviations of variance compoments from exponential decay'
    options_group.add_argument('--tau', default=0.2, type=float, help=help_msg)
    options_group.add_argument('--train_p', default=False, action='store_true',
                               help='Allow different probabilities across sites and alleles in VC prior')
    
    comp_group = parser.add_argument_group('Computational options')
    comp_group.add_argument('--gpu', default=False, action='store_true',
                            help='Use GPU-acceleration')
    comp_group.add_argument('-m', '--n_devices', default=1, type=int,
                            help='Number of GPUs to use')
    comp_group.add_argument('-s', '--partition_size', default=None, type=int,
                            help='Use kernel partitioning on GPU of this size')
    comp_group.add_argument('-n', '--n_iter', default=200, type=int,
                            help='Number of iterations for optimization(200)')
    comp_group.add_argument('-r', '--learning_rate', default=0.01, type=float,
                            help='Learning rate for optimization (0.01)')

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('-o', '--output', required=True, help='Output file')
    output_group.add_argument('-p', '--pred',
                              help='File containing sequencse for predicting genotype')

    # Parse arguments
    parsed_args = parser.parse_args()
    data_fpath = parsed_args.data
    
    kernel = parsed_args.kernel
    train_p = parsed_args.train_p
    tau = parsed_args.tau
    
    gpu = parsed_args.gpu
    n_devices = parsed_args.n_devices
    n_iter = parsed_args.n_iter
    learning_rate = parsed_args.learning_rate
    partition_size = parsed_args.partition_size

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
    y_var = data.values[:, 1] if data.shape[1] > 1 else None
      
    # Get kernel
    if kernel == 'RBF':
        kernel = ScaleKernel(RBFKernel())
    elif kernel == 'matern':
        kernel = ScaleKernel(MaternKernel())
    elif kernel == 'RQ':
        kernel = ScaleKernel(RQKernel())
    elif kernel == 'linear':
        kernel = ScaleKernel(LinearKernel())
    elif kernel == 'VC':
        n_alleles, seq_length = np.max(config['n_alleles']), config['length']
        kernel = VCKernel(n_alleles=n_alleles, seq_length=seq_length, tau=tau)
    elif kernel == 'sVC':
        n_alleles, seq_length = np.max(config['n_alleles']), config['length']
        kernel = SkewedVCKernel(n_alleles=n_alleles, seq_length=seq_length,
                                train_p=train_p, tau=tau, q=0.5)
    elif kernel == 'Diploid':
        msg = 'Diploid kernel Not implemented yet'
        raise ValueError(msg)
    else:
        msg = 'Unknown kernel provided: {}'.format(kernel)
        raise ValueError(msg)
    
    # Create model
    output_device = torch.device('cuda:0') if gpu else None
    model = EpiK(kernel, likelihood_type='Gaussian',
                 output_device=output_device, n_devices=n_devices)

    # Fit by evidence maximization
    log.write('Estimate variance components by maximizing the evidence')
    model.fit(X, y, y_var=y_var,
              n_iter=n_iter, learning_rate=learning_rate,
              partition_size=partition_size)
    
    # Write output parameters
    if hasattr(kernel, 'p'):
        ps, lambdas = kernel.p, kernel.lambdas
        if gpu:
            ps, lambdas = ps.cpu(), lambdas.cpu()
        ps = pd.DataFrame(ps.detach().numpy(), columns=np.append(alleles, '*'))
        prefix = '.'.join(out_fpath.split('.')[:-1])
        ps.to_csv('{}.p.csv'.format(prefix))
    
    if hasattr(kernel, 'lambdas'):
        lambdas =  kernel.lambdas
        if gpu:
            lambdas = lambdas.cpu()    
        prefix = '.'.join(out_fpath.split('.')[:-1])
        with open('{}.lambdas.txt'.format(prefix), 'w') as fhand:
            for l in lambdas:
                fhand.write('{}\n'.format(l))
    
    # Predict phenotype in new sequences 
    if pred_seqs.shape[0] > 0:
        log.write('Obtain phenotypic predictions')
        Xpred = seq_to_one_hot(pred_seqs, alleles=alleles)
        ypred = model.predict(Xpred)
        if gpu:
            ypred = ypred.cpu()
        result = pd.DataFrame({'ypred': ypred.detach().numpy()}, index=pred_seqs)
        result.to_csv(out_fpath)
    
    log.finish()


if __name__ == '__main__':
    main()
