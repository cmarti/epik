import numpy as np
import torch
import time
import sys
from _collections import defaultdict


def get_alleles(c, alleles=None):
        if alleles is not None:
            return(alleles)
        else:
            return(np.unique(c))

        
def seq_to_one_hot(X, alleles=None):
    m = np.array([[a for a in x] for x in X])
    onehot = []
    for i in range(m.shape[1]):
        c = m[:, i]
        for allele in get_alleles(c, alleles=alleles):
            onehot.append(get_tensor(c == allele))
    onehot = torch.stack(onehot, 1)
    return(onehot)


def one_hot_to_seq(x, alleles):
    ncol = x.shape[1]
    alpha = alleles.shape[0]
    if ncol % alpha != 0:
        raise ValueError('Check that the number of alleles is correct')
    l = int(ncol / alpha)
    a = np.hstack([alleles] * l)
    X = [''.join(a[i == 1]) for i in x]
    return(X)


def diploid_to_one_hot(X, dtype=torch.float32):
    m = torch.tensor([[int(a) for a in x] for x in X])
    onehot = torch.stack([m == 0, m == 1, m == 2], 2).to(dtype=dtype)
    return(onehot)


def to_device(tensor, output_device=None):
    if output_device is not None and output_device != -1:
        tensor = tensor.to(output_device)
    return(tensor)


def get_tensor(ndarray, dtype=torch.float32, device=None):
    if not torch.is_tensor(ndarray):
        ndarray = torch.tensor(ndarray, dtype=dtype)
    if ndarray.dtype != dtype:
        ndarray = ndarray.to(dtype=dtype)
    return(to_device(ndarray, output_device=device))


def get_gpu_memory(device=None):
    mem = torch.cuda.memory_allocated(device) / 2**20
    suffix = 'MB'
    
    if mem < 1:
        mem = mem * 2**10
        suffix = 'KB' 
    
    return('{:.0f}{}'.format(mem, suffix))


class LogTrack(object):
    def __init__(self, fhand=None):
        if fhand is None:
            fhand = sys.stderr
        self.fhand = fhand
        self.start = time.time()

    def write(self, msg, add_time=True):
        if add_time:
            msg = '[ {} ] {}\n'.format(time.ctime(), msg)
        else:
            msg += '\n'
        self.fhand.write(msg)
        self.fhand.flush()

    def finish(self):
        t = time.time() - self.start
        self.write('Finished succesfully. Time elapsed: {:.1f} s'.format(t))
        

def guess_space_configuration(seqs):
    """
    Guess the sequence space configuration from a collection of sequences
    This allows to have different number of alleles per site and maintain 
    the order in which alleles appear in the sequences when enumerating the 
    alleles per position
    
    Parameters
    ----------
    seqs: array-like of shape (n_genotypes,)
        Vector or list containing the sequences from which we want to infer
        the space configuration
        
       
    Returns
    -------
    config: dict with keys {'seq_length', 'n_alleles', 'alphabet'}
            Returns a dictionary with the inferred configuration of the discrete
            space where the sequences come from.
    
    """
    
    alleles = defaultdict(dict)
    for seq in seqs:
        for i, a in enumerate(seq):
            alleles[i][a] = 1 
    length = len(alleles)
    config = {'seq_length': length,
              'n_alleles': [len(alleles[i]) for i in range(length)],
              'alphabet': [[a for a in alleles[i].keys()] for i in range(length)]}
    return(config)


def split_training_test(X, y, y_var=None, ptrain=0.8, dtype=None):
    ps = np.random.uniform(size=X.shape[0])
    train = ps <= ptrain
    train_x, train_y = X[train, :], y[train]
    
    if y_var is None:
        train_y_var = None
    else:
        train_y_var = y_var[train] 

    test = ps > ptrain
    test_x, test_y = X[test, :], y[test]
    
    output = [train_x, train_y, test_x, test_y, train_y_var]
    if dtype is not None:
        output = [get_tensor(a, dtype=dtype) if a is not None else None
                  for a in output]
    return(output)


def ps_to_variances(ps):
    v = 1 / ps
    v = (v.T / v.sum(1)).T
    return(v)


def get_full_space_one_hot(seq_length, n_alleles, dtype=torch.float32):
    n = n_alleles ** seq_length
    i = torch.arange(n)
    
    c = i
    one_hot = []
    for _ in range(seq_length):
        r = c % n_alleles
        for j in range(n_alleles):
            one_hot.append(r == j)
        c = torch.div(c, n_alleles, rounding_mode='floor')
    X = torch.vstack(one_hot).T.to(dtype=dtype)
    return(X)


def log1mexp(x):
    """Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    """
    two = torch.tensor([2.], dtype=x.dtype).to(device=x.device)
    mask = -torch.log(two) < x  # x < 0
    return(torch.where(mask, (-x.expm1()).log(), (-x.exp()).log1p()))

