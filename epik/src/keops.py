import numpy as np
import torch as torch

from _functools import partial
from itertools import combinations

from scipy.special._basic import comb
from torch.nn import Parameter
from pykeops.torch import LazyTensor
from gpytorch.kernels.keops.keops_kernel import KeOpsKernel
from linear_operator.operators import KernelLinearOperator


class Kernel(KeOpsKernel):
    def register_params(self, params={}, constraints={}):
        self.params = params
        for param_name, param in params.items():
            self.register_parameter(name=param_name, parameter=param)
            
        for param_name, constraint in constraints.items():
            self.register_constraint(param_name, constraint)


class RhoPiKernel(Kernel):
    def __init__(self, n_alleles, seq_length, train_p=True,
                 logit_rho0=None, log_p0=None, **kwargs):
        self.alpha = n_alleles
        self.l = seq_length
        self.t = self.l * self.alpha
        self.train_p = train_p
        self.log_p0 = log_p0
        self.logit_rho0 = logit_rho0
        super().__init__(**kwargs)
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


################################################################################
#
# class LambdasKernel(SequenceKernel):
#     def calc_polynomial_coeffs(self):
#         lambdas = self.lambdas_p
#
#         B = np.zeros((self.s, self.s))
#         idx = np.arange(self.s)
#         for k in idx:
#             k_idx = idx != k
#             k_lambdas = lambdas[k_idx]
#             norm_factor = 1 / np.prod(k_lambdas - lambdas[k])
#
#             for power in idx:
#                 lambda_combs = list(combinations(k_lambdas, self.l - power))
#                 p = np.sum([np.prod(v) for v in lambda_combs])
#                 B[power, k] = (-1) ** (power) * p * norm_factor
#
#         return(B)
#
#     @property
#     def lambdas(self):
#         log_lambdas = self.lambdas_prior.theta_to_log_lambdas(self.raw_theta, kernel=self)
#         lambdas = torch.exp(log_lambdas)
#         return(lambdas)
#
#     def set_lambdas_prior(self, lambdas_prior):
#         if lambdas_prior is None:
#             lambdas_prior = LambdasFlatPrior(seq_length=self.l)
#         self.lambdas_prior = lambdas_prior
#         self.lambdas_prior.set(self)
#
#
# class VarianceComponentKernel(LambdasKernel):
#     is_stationary = True
#     def __init__(self, n_alleles, seq_length, lambdas_prior=None,
#                  **kwargs):
#         super().__init__(n_alleles, seq_length, **kwargs)
#
#         self.define_aux_variables()
#         self.set_lambdas_prior(lambdas_prior)
#
#     def define_aux_variables(self):
#         self.krawchouk_matrix = Parameter(self.calc_krawchouk_matrix(),
#                                           requires_grad=False)
#         self.d_powers_matrix = Parameter(self.calc_d_powers_matrix(),
#                                          requires_grad=False)
#
#     def calc_w_kd(self, k, d):
#         ss = 0
#         for q in range(self.s):
#             ss += (-1.)**q * (self.alpha-1.)**(k-q) * comb(d,q) * comb(self.l-d,k-q)
#         return(ss / self.n)
#
#     def calc_d_powers_matrix(self):
#         p = np.arange(self.s).reshape(1, self.s)
#         d = np.arange(self.s).reshape(self.s, 1)
#         d_powers = d ** p
#         return(get_tensor(d_powers))
#
#     def calc_krawchouk_matrix(self):
#         w_kd = np.zeros((self.s, self.s))
#         for k in range(self.s):
#             for d in range(self.s):
#                 w_kd[d, k] = self.calc_w_kd(k, d)
#         return(get_tensor(w_kd))
#
#     def _covar_func(self, x1, x2, lambdas, **kwargs):
#         dtype = x1.dtype
#         x1, x2 = LazyTensor(x1[None, :, None, :]), LazyTensor(x2[None, None, :, :])
#         hamming_dist = self.l - (x1 * x2).sum(-1)
#         exponents = LazyTensor(torch.arange(self.s).to(dtype=dtype).reshape(self.s, 1, 1, 1))
#         print(hamming_dist)
#         print(exponents)
#         d_powers =  hamming_dist.power(exponents) 
#         print(d_powers)
#
#         w_d = self.krawchouk_matrix @ lambdas
#         c_d = torch.linalg.solve(self.d_powers_matrix, w_d)
#         print(c_d)
#         c_d = LazyTensor(c_d.reshape(self.s, 1, 1, 1))
#         print(c_d)
#         kernel = (d_powers * c_d).sum(-1)
#         print(kernel)
#         return(kernel)
#
#     def _keops_forward(self, x1, x2, **kwargs):
#         covar_func = partial(self._covar_func, lambdas=self.lambdas)
#         return KernelLinearOperator(x1, x2, covar_func=covar_func, **kwargs)
#
#     def get_params(self):
#         return({'lambdas': self.lambdas})
#
#
# class DeltaPKernel(VarianceComponentKernel):
#     def __init__(self, n_alleles, seq_length, P, **kwargs):
#         lambdas_prior = LambdasDeltaPrior(seq_length, n_alleles, P=P)
#         super().__init__(n_alleles, seq_length, lambdas_prior=lambdas_prior,
#                          **kwargs)
