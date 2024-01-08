import numpy as np
import torch as torch

from scipy.special._basic import comb
from torch.nn import Parameter
from pykeops.torch import LazyTensor
from gpytorch.kernels.keops.keops_kernel import KeOpsKernel
from linear_operator.operators import KernelLinearOperator
from epik.src.utils import log1mexp


class SequenceKernel(KeOpsKernel):
    def __init__(self, n_alleles, seq_length, **kwargs):
        self.alpha = n_alleles
        self.l = seq_length
        self.s = self.l + 1
        self.t = self.l * self.alpha
        super().__init__(**kwargs)
        
    def register_params(self, params={}, constraints={}):
        self.params = params
        for param_name, param in params.items():
            self.register_parameter(name=param_name, parameter=param)
            
        for param_name, constraint in constraints.items():
            self.register_constraint(param_name, constraint)
            

class RhoPiKernel(SequenceKernel):
    def __init__(self, n_alleles, seq_length, train_p=True,
                 logit_rho0=None, log_p0=None, **kwargs):
        super().__init__(n_alleles, seq_length, **kwargs)
        self.train_p = train_p
        self.log_p0 = log_p0
        self.logit_rho0 = logit_rho0
        self.set_params()
    
    def zeros_like(self, x):
        return(torch.zeros(x.shape).to(dtype=x.dtype, device=x.device))

    def set_params(self):
        log_p0 = -torch.ones((self.l, self.alpha)) if self.log_p0 is None else self.log_p0
        logit_rho0 = torch.full((self.l, 1), np.log(np.exp(1 / self.l) - 1)) if self.logit_rho0 is None else self.logit_rho0 
        params = {'logit_rho': Parameter(logit_rho0, requires_grad=True),
                  'log_p': Parameter(log_p0, requires_grad=self.train_p)}
        self.register_params(params)
    
    def get_factor(self):
        log1mrho = torch.logaddexp(self.zeros_like(self.logit_rho), self.logit_rho)
        log_rho = self.logit_rho + log1mrho
        log_p = self.log_p - torch.logsumexp(self.log_p, axis=1).unsqueeze(1)
        log_eta = log1mexp(log_p) - log_p 
        log_one_p_eta_rho = torch.logaddexp(self.zeros_like(log_rho), log_rho + log_eta)
        factor = log_one_p_eta_rho - log1mrho
        return(torch.sqrt(factor.reshape(1, self.t)))
    
    def get_c(self):
        return(-torch.logaddexp(self.zeros_like(self.logit_rho), self.logit_rho).sum())
    
    def _nonkeops_forward(self, x1, x2, diag=False, **kwargs):
        f = self.get_factor()
        c = self.get_c()
        x1_ = x1[..., :, None, :] * f
        x2_ = x2[..., None, :, :] * f
        return(torch.exp(c + (x1_ * x2_).sum(-1)))

    def _covar_func(self, x1, x2, **kwargs):
        x1_ = LazyTensor(x1[..., :, None, :])
        x2_ = LazyTensor(x2[..., None, :, :])
        K = (x1_ * x2_).sum(-1).exp()
        return(K)
    
    def _keops_forward(self, x1, x2, **kwargs):
        f = self.get_factor()
        c = torch.exp(self.get_c())
        x1_, x2_ = x1 * f, x2 * f
        return(c * KernelLinearOperator(x1_, x2_, covar_func=self._covar_func, **kwargs))
    

class RhoKernel(RhoPiKernel):
    def __init__(self, n_alleles, seq_length, logit_rho0=None, **kwargs):
        super().__init__(n_alleles, seq_length, logit_rho0=logit_rho0,
                         train_p=False, **kwargs)


class VarianceComponentKernel(SequenceKernel):
    is_stationary = True
    def __init__(self, n_alleles, seq_length, log_lambdas0=None, **kwargs):
        super().__init__(n_alleles, seq_length, **kwargs)
        self.n = self.alpha ** self.l
        self.log_lambdas0 = log_lambdas0
        self.set_params()
        
    def log_factorial(self, x):
        return(torch.lgamma(x+1))
    
    def log_comb(self, n, k):
        '''
        from https://stackoverflow.com/questions/4775029/finding-combinatorial-of-large-numbers
        lngamma(n+1) - lngamma(k+1) - lngamma(n-k+1)
        '''
        return(self.log_factorial(n) - self.log_factorial(k) - self.log_factorial(n-k))
    
    def set_params(self):
        log_lambdas0 = -torch.arange(self.s).to(dtype=torch.float) if self.log_lambdas0 is None else self.log_lambdas0
        log_lambdas0 = log_lambdas0.reshape((self.s, 1))
        w_kd = self.calc_krawchouk_matrix()
        d_powers_inv = self.calc_d_powers_matrix_inv()
        k = torch.arange(self.s).to(dtype=torch.float).reshape(1, 1, self.s)
        params = {'log_lambdas': Parameter(log_lambdas0, requires_grad=True),
                  'w_kd': Parameter(w_kd, requires_grad=False),
                  'd_powers_inv': Parameter(d_powers_inv, requires_grad=False),
                  'k': Parameter(k, requires_grad=False)}
        self.register_params(params)
        
    def calc_krawchouk_matrix(self):
        d = torch.arange(self.s).reshape((self.s, 1, 1))
        k = torch.arange(self.s).reshape((1, self.s, 1))
        q = torch.arange(self.s).reshape((1, 1, self.s))
        w_kd = ((-1.)**q * (self.alpha-1.)**(k-q) * comb(d,q) * comb(self.l-d,k-q)).sum(-1) / self.n
        return(w_kd.to(dtype=torch.float))
    
    def get_w_d(self):
        '''calculates the covariance for each hamming distance'''
        lambdas = torch.exp(self.log_lambdas)
        w_d = self.w_kd @ lambdas
        return(w_d.reshape((1, 1, self.s)))

    def calc_d_powers_matrix_inv(self):
        p = torch.arange(self.s).reshape(1, self.s)
        d = torch.arange(self.s).reshape(self.s, 1)
        d_powers = (d ** p).to(dtype=torch.float)
        return(torch.linalg.inv(d_powers))
    
    def get_c_d(self):
        '''calculates coefficients for the covariance as a polynomial in d'''
        lambdas = torch.exp(self.log_lambdas)
        c_d = self.d_powers_inv @ (self.w_kd @ lambdas)
        return(c_d)

    def get_k(self):
        return(self.k)
    
    def _nonkeops_forward(self, x1, x2, diag=False, **kwargs):
        x1_ = x1[..., :, None, :]
        x2_ = x2[..., None, :, :]
        d = self.l - (x1_ * x2_).sum(-1).unsqueeze(-1)
        w_d = self.get_w_d()
        k = ((d == self.k) * w_d).sum(-1)
        return(k)
    
    def _nonkeops_forward_polynomial_d(self, x1, x2):
        x1_ = x1[..., :, None, :]
        x2_ = x2[..., None, :, :]
        d = self.l - (x1_ * x2_).sum(-1).unsqueeze(-1)
        d_powers =  d ** self.k
        c_d = self.get_c_d().reshape((1, 1, self.s))
        k = (d_powers * c_d).sum(-1)
        return(k)

    def _covar_func(self, x1, x2, **kwargs):
        x1_ = LazyTensor(x1[..., :, None, :])
        x2_ = LazyTensor(x2[..., None, :, :])
        d = self.l - (x1_ * x2_).sum(-1)
        
        c_d = self.get_c_d()
        k = c_d[0]
        d_power = 1
        for i in range(1, self.s):
            d_power = d_power * d
            k += c_d[i] * d_power 
        return(k)
        
    def _keops_forward(self, x1, x2, **kwargs):
        return(KernelLinearOperator(x1, x2, covar_func=self._covar_func, **kwargs))


class DeltaKernel(VarianceComponentKernel):
    is_stationary = True
    def __init__(self, n_alleles, seq_length, P,
                 log_lambdas0=None, log_a0=None, **kwargs):
        super().__init__(n_alleles, seq_length, **kwargs)
        self.n = self.alpha ** self.l
        self.P = P
        self.log_a0 = log_a0 
        self.log_lambdas0 = log_lambdas0
        self.set_params()
    
    def set_params(self):
        log_lambdas0 = -torch.arange(self.s) if self.log_lambdas0 is None else self.log_lambdas0
        log_lambdas0 = log_lambdas0.reshape((self.s, 1))
        w_kd = self.calc_krawchouk_matrix()
        d_powers_inv = self.calc_d_powers_matrix_inv()
        k = torch.arange(self.s).to(dtype=torch.float).reshape(1, 1, self.s)
        params = {'log_lambdas': Parameter(log_lambdas0, requires_grad=True),
                  'w_kd': Parameter(w_kd, requires_grad=False),
                  'd_powers_inv': Parameter(d_powers_inv, requires_grad=False),
                  'k': Parameter(k, requires_grad=False)}
        self.register_params(params)


class AdditiveKernel(SequenceKernel):
    def __init__(self, n_alleles, seq_length, log_lambdas0=None, **kwargs):
        super().__init__(n_alleles, seq_length, **kwargs)
        self.log_lambdas0 = log_lambdas0
        self.logn = seq_length * np.log(n_alleles)
        self.set_params()
    
    def get_matrix(self):
        m = torch.tensor([[1, -self.l],
                          [0., self.alpha]])
        return(m)  
    
    def set_params(self):
        log_lambdas0 = torch.tensor([0., 0.]) if self.log_lambdas0 is None else self.log_lambdas0
        params = {'log_lambdas': Parameter(log_lambdas0, requires_grad=True),
                  'm': Parameter(self.get_matrix(), requires_grad=False)}
        self.register_params(params)
    
    def lambdas_to_coeffs(self, lambdas):
        coeffs = self.m @ lambdas
        return(coeffs)

    def get_coeffs(self):
        return(self.lambdas_to_coeffs(torch.exp(self.log_lambdas)))
    
    def _nonkeops_forward(self, x1, x2, diag=False, **kwargs):
        coeffs = self.get_coeffs()
        x1_ = x1[..., :, None, :]
        x2_ = x2[..., None, :, :]
        return(coeffs[0] + coeffs[1] * (x1_ * x2_).sum(-1))

    def _covar_func(self, x1, x2, **kwargs):
        coeffs = self.get_coeffs()
        b = torch.sqrt(coeffs[1])
        x1_ = LazyTensor(x1[..., :, None, :] * b)
        x2_ = LazyTensor(x2[..., None, :, :] * b)
        K = (x1_ * x2_).sum(-1) + coeffs[0]
        return(K)
    
    def _keops_forward(self, x1, x2, **kwargs):
        return(KernelLinearOperator(x1, x2, covar_func=self._covar_func, **kwargs))
