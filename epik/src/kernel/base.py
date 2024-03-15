import numpy as np
import torch as torch

from pykeops.torch import LazyTensor
from linear_operator.operators import MatmulLinearOperator
from torch.nn import Parameter
from gpytorch.settings import max_cholesky_size
from gpytorch.kernels.kernel import Kernel
from gpytorch.lazy.lazy_tensor import delazify


class SequenceKernel(Kernel):
    def __init__(self, n_alleles, seq_length, binary=False,
                 dtype=torch.float32, use_keops=False, **kwargs):
        self.alpha = n_alleles

        if binary and n_alleles != 2:
            raise ValueError('binary encoding can only be used with 2 alleles')
        
        self.binary = binary
        self.l = seq_length
        self.s = self.l + 1
        self.t = self.l * self.alpha
        self.fdtype = dtype
        self.use_keops = use_keops
        self.logn = self.l * np.log(self.alpha)
        self.logam1 = np.log(self.alpha - 1)
        super().__init__(**kwargs)
        
    def zeros_like(self, x):
        return(torch.zeros(x.shape).to(dtype=x.dtype, device=x.device))
    
    def register_params(self, params={}, constraints={}):
        self.params = params
        for param_name, param in params.items():
            self.register_parameter(name=param_name, parameter=param)
            
        for param_name, constraint in constraints.items():
            self.register_constraint(param_name, constraint)
    
    def inner_product(self, x1, x2, metric=None, diag=False):
        if metric is None:
            metric = torch.eye(x2.shape[1], dtype=x2.dtype, device=x2.device)
            
        if diag:
            min_size = min(x1.shape[0], x2.shape[0])
            return((torch.matmul(x1[:min_size, :], metric) * x2[:min_size, :]).sum(1))
        else:
            return(torch.matmul(x1, torch.matmul(metric, x2.T)))
    
    def s_to_d(self, s):
        if self.binary:
            d = self.l / 2. - 0.5 * s
        else:
            d = float(self.l) - s
        return(d)

    def calc_hamming_distance(self, x1, x2):
        s = self.inner_product(x1, x2)
        d = self.s_to_d(s)
        return(d)
    
    def calc_hamming_distance_linop(self, x1, x2):
        if self.binary:
            a = torch.tensor([[self.l / 2.0]], device=x1.device, dtype=x1.dtype)
            d = a - MatmulLinearOperator(x1 / np.sqrt(2), x2.T / np.sqrt(2))
        else:
            a = torch.tensor([[float(self.l)]], device=x1.device, dtype=x1.dtype)
            d = a - MatmulLinearOperator(x1, x2.T)
        return(d)
    
    def calc_hamming_distance_keops(self, x1, x2):
        if self.binary:
            x1_ = LazyTensor(x1[..., :, None, :] / np.sqrt(2))
            x2_ = LazyTensor(x2[..., None, :, :] / np.sqrt(2))
            s = (x1_ * x2_).sum(-1)
            d = self.l / 2. - s
        else:
            x1_ = LazyTensor(x1[..., :, None, :])
            x2_ = LazyTensor(x2[..., None, :, :])
            s = (x1_ * x2_).sum(-1)
            d = float(self.l) - s
        return(d)
    
    def _constant_linop(self, x1, x2):
        z1 = torch.ones((x1.shape[0], 1), device=x1.device, dtype=x1.dtype)
        z2 = torch.ones((1, x2.shape[0]), device=x2.device, dtype=x2.dtype)
        c = MatmulLinearOperator(z1, z2)
        return(c)
    
    def forward(self, x1, x2, diag=False, **kwargs):
        if diag:
            kernel = self._nonkeops_forward(x1, x2, diag=True, **kwargs)
        else:
            max_size = max_cholesky_size.value()
            if self.use_keops or (x1.size(-2) > max_size or x2.size(-2) > max_size):
                kernel = self._keops_forward(x1, x2, **kwargs)
            else:
                try:
                    kernel = self._nonkeops_forward(x1, x2, diag=False, **kwargs)
                except RuntimeError:
                    torch.cuda.empty_cache()
                    kernel = self._keops_forward(x1, x2, **kwargs)
        return(kernel)
    

class AdditiveHeteroskedasticKernel(SequenceKernel):
    @property
    def is_stationary(self) -> bool:
        return self.base_kernel.is_stationary

    def __init__( self, base_kernel, n_alleles=None, seq_length=None,
                  log_ds0=None, a=0.5, **kwargs):
        if base_kernel.active_dims is not None:
            kwargs["active_dims"] = base_kernel.active_dims
        
        if hasattr(base_kernel, 'alpha'):
            n_alleles = base_kernel.alpha
        else:
            if n_alleles is None:
                msg = 'If the base kernel does not have n_alleles attribute, '
                msg += 'it should be provided'
                raise ValueError(msg)
        
        if hasattr(base_kernel, 'l'):
            seq_length = base_kernel.l
        else:
            if seq_length is None:
                msg = 'If the base kernel does not have seq_length attribute, '
                msg += 'it should be provided'
                raise ValueError(msg)
        
        super().__init__(n_alleles, seq_length, **kwargs)
        self.log_ds0 = log_ds0
        self.a = a
        self.set_params()
        self.base_kernel = base_kernel

    def set_params(self):
        theta = torch.zeros((self.l, self.alpha)) if self.log_ds0 is None else self.log_ds0
        params = {'theta': Parameter(theta, requires_grad=True),
                  'theta0': Parameter(5 * torch.ones((1,)), requires_grad=True)}
        self.register_params(params)
        
    def get_theta(self):
        t = self.theta
        return(t - t.mean(1).unsqueeze(1))
    
    def get_theta0(self):
        return(self.theta0)
    
    def f(self, x, theta0, theta, a=0, b=1):
        phi = theta0 + (x * theta.reshape(1, 1, self.l * self.alpha)).sum(-1)
        r = a + (b - a) * torch.exp(phi) / (1 + torch.exp(phi))
        return(r)
    
    def forward(self, x1, x2, last_dim_is_batch=False, diag=False, **params):
        orig_output = self.base_kernel.forward(x1, x2, diag=diag,
                                               last_dim_is_batch=last_dim_is_batch,
                                               **params)
        theta0, theta = self.get_theta0(), self.get_theta()
        f1 = self.f(x1, theta0, theta, a=self.a).T
        f2 = self.f(x2, theta0, theta, a=self.a)
        
        if last_dim_is_batch:
            f1 = f1.unsqueeze(-1)
            f2 = f2.unsqueeze(-1)
        if diag:
            f1 = f1.unsqueeze(-1)
            f2 = f2.unsqueeze(-1)
            return(f1 * f2 * delazify(orig_output))
        else:
            return(f1 * f2 * orig_output)

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
        return self.base_kernel.prediction_strategy(train_inputs, train_prior_dist, train_labels, likelihood)
