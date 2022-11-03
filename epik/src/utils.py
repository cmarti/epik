import numpy as np
import torch


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
    return(to_device(ndarray, output_device=device))


def get_gpu_memory(device=None):
    mem = torch.cuda.memory_allocated(device) / 2**20
    suffix = 'MB'
    
    if mem < 1:
        mem = mem * 2**10
        suffix = 'KB' 
    
    return('{:.0f}{}'.format(mem, suffix))
