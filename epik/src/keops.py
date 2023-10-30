import torch as torch

from scipy.special._basic import comb
from torch.nn import Parameter
from pykeops.torch import LazyTensor
from gpytorch.kernels.keops.keops_kernel import KeOpsKernel
from linear_operator.operators import KernelLinearOperator


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
            

class HetRBFKernel(SequenceKernel):
    def __init__(self, n_alleles, seq_length, 
                 log_lengthscale0=None, log_ds0=None, **kwargs):
        super().__init__(n_alleles, seq_length, **kwargs)
        self.log_ds0 = log_ds0
        self.log_lengthscale0 = log_lengthscale0
        self.set_params()

    def set_params(self):
        log_ds0 = torch.zeros((1, 1, self.l * self.alpha)) if self.log_ds0 is None else self.log_ds0
        log_lengthscale0 = torch.zeros((1,)) if self.log_lengthscale0 is None else self.log_lengthscale0 
        params = {'log_lengthscale': Parameter(log_lengthscale0, requires_grad=True),
                  'log_ds': Parameter(log_ds0, requires_grad=True)}
        self.register_params(params)
        
    def get_ds(self):
        print(self.log_ds)
        return(torch.exp(self.log_ds) * self.get_lengthscale())
    
    def get_lengthscale(self):
        print(self.log_lengthscale)
        return(torch.exp(self.log_lengthscale))
    
    def _nonkeops_forward(self, x1, x2, diag=False, **kwargs):
        ds = self.get_ds()
        l = self.get_lengthscale()
        x1_ = x1[..., :, None, :]
        x2_ = x2[..., None, :, :]
        d = self.l - (x1_ * x2_).sum(-1)
        d1 = (x1_ * ds).sum(-1)
        d2 = (x2_ * ds).sum(-1)
        k = d1 * d2 * torch.exp(-l * d)
        return(k)

    def _covar_func(self, x1, x2, **kwargs):
        ds = self.get_ds()
        x1_ = LazyTensor(x1[..., :, None, :])
        x2_ = LazyTensor(x2[..., None, :, :])
        k = ((x1_ * ds).sum(-1) + (x2_ * ds).sum(-1) + (x1_ * x2_).sum(-1)).exp()
        return(k)
    
    def _keops_forward(self, x1, x2, **kwargs):
        l = self.get_lengthscale()
        x1_ = x1 / l
        x2_ = x2 / l
        return(KernelLinearOperator(x1_, x2_, covar_func=self._covar_func, **kwargs))


class RhoPiKernel(SequenceKernel):
    def __init__(self, n_alleles, seq_length, train_p=True,
                 logit_rho0=None, log_p0=None, **kwargs):
        super().__init__(n_alleles, seq_length, **kwargs)
        self.train_p = train_p
        self.log_p0 = log_p0
        self.logit_rho0 = logit_rho0
        self.set_params()

    def set_params(self):
        log_p0 = -torch.ones((self.l, self.alpha)) if self.log_p0 is None else self.log_p0
        logit_rho0 = torch.zeros((self.l, 1)) if self.logit_rho0 is None else self.logit_rho0 
        params = {'logit_rho': Parameter(logit_rho0, requires_grad=True),
                  'log_p': Parameter(log_p0, requires_grad=self.train_p)}
        self.register_params(params)
        
    def get_factor(self):
        rho = torch.exp(self.logit_rho) / (1 + torch.exp(self.logit_rho))
        log_p = self.log_p - torch.logsumexp(self.log_p, axis=1).unsqueeze(1)
        p = torch.exp(log_p)
        eta = (1 - p) / p
        factor = torch.log(1 + eta * rho) - torch.log(1 - rho)
        return(torch.sqrt(factor.reshape(1, self.t)))
    
    def get_c(self):
        rho = torch.exp(self.logit_rho) / (1 + torch.exp(self.logit_rho))
        return(torch.log(1 - rho).sum())
    
    def _nonkeops_forward(self, x1, x2, diag=False, **kwargs):
        f = self.get_factor()
        c = self.get_c()
        x1_ = x1[..., :, None, :] * f
        x2_ = x2[..., None, :, :] * f
        return(torch.exp(c + (x1_ * x2_).sum(-1)))

    def _covar_func(self, x1, x2, **kwargs):
        c = self.get_c()
        x1_ = LazyTensor(x1[..., :, None, :])
        x2_ = LazyTensor(x2[..., None, :, :])
        K = (c + (x1_ * x2_).sum(-1)).exp()
        return(K)
    
    def _keops_forward(self, x1, x2, **kwargs):
        f = self.get_factor()
        x1_, x2_ = x1 * f, x2 * f
        return(KernelLinearOperator(x1_, x2_, covar_func=self._covar_func, **kwargs))
    

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
    
    def _nonkeops_forward(self, x1, x2):
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

