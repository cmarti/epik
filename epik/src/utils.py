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
    config: dict with keys {'length', 'n_alleles'}
            Returns a dictionary with the inferred configuration of the discrete
            space where the sequences come from.
    
    """
    
    alleles = defaultdict(dict)
    for seq in seqs:
        for i, a in enumerate(seq):
            alleles[i][a] = 1 
    length = len(alleles)
    config = {'length': length,
              'n_alleles': [len(alleles[i]) for i in range(length)],
              'alphabet': [[a for a in alleles[i].keys()] for i in range(length)]}
    return(config)
