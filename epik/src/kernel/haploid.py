import numpy as np
import torch as torch

from functools import partial
from itertools import combinations
from scipy.special._basic import comb
from torch.nn import Parameter
from pykeops.torch import LazyTensor
from linear_operator.operators import (KernelLinearOperator, MatmulLinearOperator,
                                       DiagLinearOperator)

from epik.src.utils import get_tensor, log1mexp
from epik.src.kernel.base import SequenceKernel
from epik.src.priors import (LambdasFlatPrior, LambdasDeltaPrior, RhosPrior,
                             AllelesProbPrior)


def calc_d_powers_inverse(l, max_k=None):
    if max_k is None:
        max_k = l
    q = torch.arange(max_k + 1).unsqueeze(1)
    d = torch.arange(l + 1).unsqueeze(0)
    log_distances = torch.log(d)

    norm_factors = torch.prod((d - q).fill_diagonal_(1), axis=0)
    print(norm_factors)
    print(torch.exp(log_comb(torch.tensor([l]), d)))


def calc_vandermonde_inverse(values, n=None):
        s = values.shape[0]
        if n is None:
            n = s
        
        B = np.zeros((n, s))
        idx = np.arange(s)
        sign = np.zeros((n, s))

        for k in range(s):
            v_k = values[idx != k]
            norm_factor = 1 / np.prod(v_k - values[k])
        
            for power in range(n):
                v_k_combs = list(combinations(v_k, s - power - 1))
                p = np.sum([np.prod(v) for v in v_k_combs])
                B[power, k] = sign[power, k] * p * norm_factor
                sign[power, k] = (-1) ** (power)
                print(1/norm_factor, p)
        return(B)

class AdditiveKernel(SequenceKernel):
    def __init__(self, n_alleles, seq_length,
                 log_lambdas0=None, log_var0=None,
                 **kwargs):
        super().__init__(n_alleles, seq_length, **kwargs)
        self.log_lambdas0 = log_lambdas0
        self.log_var0 = log_var0
        self.set_params()
    
    def get_matrix(self):
        m = torch.tensor([[1, -self.l],
                          [0., self.alpha]])
        return(m)  
    
    def set_params(self):
        m = self.get_matrix()

        if self.log_lambdas0 is None: 
            # log_var0 = 0. if self.log_var0 is None else self.log_var0
            # var0 = np.exp(log_var0)
            # coeffs0 = torch.tensor([0.5 * var0, var0 /(2 * self.l)]).to(dtype=self.dtype)
            # log_lambdas0 = torch.log(torch.linalg.solve(m, coeffs0))
            log_lambdas0 = torch.tensor([0., 0.], dtype=self.dtype)
        else:
            log_lambdas0 = self.log_lambdas0.to(dtype=self.dtype)

        params = {'log_lambdas': Parameter(log_lambdas0, requires_grad=True),
                  'm': Parameter(m, requires_grad=False)}
        self.register_params(params)
    
    def lambdas_to_coeffs(self, lambdas):
        coeffs = self.m @ lambdas
        return(coeffs)

    def get_coeffs(self):
        return(self.lambdas_to_coeffs(torch.exp(self.log_lambdas)))
    
    def forward(self, x1, x2, diag=False, **kwargs):
        coeffs = self.get_coeffs()
        # print(self.log_lambdas, coeffs.detach().cpu().numpy())
        s = MatmulLinearOperator(x1, x2.T)
        x1_ = torch.ones(size=(x1.shape[0], 1), dtype=coeffs.dtype, device=coeffs.device)
        x2_ = torch.ones(size=(1, x2.shape[0]), dtype=coeffs.dtype, device=coeffs.device)
        s0 = MatmulLinearOperator(x1_, x2_)
        if diag:
            return(coeffs[0] * x1_ + coeffs[1] * s.diagonal(dim1=-1, dim2=-2))
        else:
            return(coeffs[0] *  s0 + coeffs[1] * s)
        
class _LambdasKernel(SequenceKernel):
    def calc_polynomial_coeffs(self):
        lambdas = self.lambdas_p
        
        B = np.zeros((self.s, self.s))
        idx = np.arange(self.s)
        for k in idx:
            k_idx = idx != k
            k_lambdas = lambdas[k_idx]
            norm_factor = 1 / np.prod(k_lambdas - lambdas[k])
        
            for power in idx:
                lambda_combs = list(combinations(k_lambdas, self.l - power))
                p = np.sum([np.prod(v) for v in lambda_combs])
                B[power, k] = (-1) ** (power) * p * norm_factor

        return(B)

def log_factorial(x):
    return(torch.lgamma(x + 1))

def log_comb(n, k):
    return(log_factorial(n) - log_factorial(k) - log_factorial(n - k))


class VarianceComponentKernel(_LambdasKernel):
    is_stationary = True
    def __init__(self, n_alleles, seq_length, log_lambdas0=None,
                 log_var0=2., max_k=None, **kwargs):
        super().__init__(n_alleles, seq_length, **kwargs)
        self.max_k = self.l if max_k is None else min(max_k, self.l)
        self.log_lambdas0 = log_lambdas0
        self.log_var0 = log_var0
        self.set_params()

    def calc_d_k_q(self, s, max_k):
        if max_k is None:
            max_k = s
        d = torch.arange(s).reshape((s, 1, 1))
        k = torch.arange(max_k + 1).reshape((1, max_k + 1, 1))
        q = torch.arange(max_k + 1).reshape((1, 1, max_k + 1))
        return(d, k, q)
    
    def calc_log_w_kd(self):
        d, k, q = self.calc_d_k_q(self.s, self.max_k)
        log_w_kd = (k-q) * self.logam1 + log_comb(d, q)
        log_w_kd += log_comb(self.l-d, k-q) - self.logn
        sign = (-1.)**q
        return(sign, log_w_kd)
    
    def calc_w_d(self, w_kd_sign, log_w_kd, log_lambdas):
        log_w_d_ = log_w_kd + log_lambdas.reshape((1, log_lambdas.shape[0], 1))
        w_d = (w_kd_sign * torch.exp(log_w_d_)).sum((1, 2))
        return(w_d)

    def calc_d_powers_matrix_inv(self):
        d = torch.arange(self.l + 1).unsqueeze(0).to(dtype=torch.float64)
        A = d.T ** d
        d_powers_inv = torch.linalg.inv(A)[:self.max_k + 1]
        return(d_powers_inv)
    
    def calc_d_polynomial_coeffs_matrix_log(self, w_kd_sign, log_w_kd):
        d_powers_inv = self.calc_d_powers_matrix_inv()
        n, m = d_powers_inv.shape
        d_powers_inv = d_powers_inv.reshape((n, m, 1, 1))
        d_powers_inv_sign = torch.sign(d_powers_inv)
        d_powers_inv_log = torch.log(torch.abs(d_powers_inv))
        log_c_p = d_powers_inv_log + log_w_kd.unsqueeze(0)
        sign = w_kd_sign * d_powers_inv_sign
        return(sign, log_c_p)
    
    def calc_c_d(self, c_p_sign, log_c_p, log_lambdas):
        log_lambdas = log_lambdas.reshape((1, 1, self.max_k + 1, 1))
        c_d = (c_p_sign * torch.exp(log_c_p + log_lambdas)).sum((1, 2, 3)).flatten()
        return(c_d)
    
    def calc_log_lambdas0(self, c_p_sign, log_c_p):
        if self.log_lambdas0 is None:
            # Match linearly decaying correlations with specified variance
            exp_coeffs = torch.zeros((log_c_p.shape[0], 1))
            exp_coeffs[0] = np.exp(self.log_var0)
            exp_coeffs[1] = -np.exp(self.log_var0) / self.l
            c_p = (c_p_sign * torch.exp(log_c_p)).sum((1, 3))
            log_lambdas0 = np.log(torch.linalg.solve_triangular(c_p, exp_coeffs, upper=True)+1e-10)
        else:
            log_lambdas0 = self.log_lambdas0
        log_lambdas0 = log_lambdas0.reshape((self.max_k + 1, 1))
        return(log_lambdas0)

    def set_params(self):
        w_kd_sign, log_w_kd = self.calc_log_w_kd()
        c_p_sign, log_c_p = self.calc_d_polynomial_coeffs_matrix_log(w_kd_sign, log_w_kd)
        log_lambdas0 = self.calc_log_lambdas0(c_p_sign, log_c_p)

        params = {'log_lambdas': Parameter(log_lambdas0.to(dtype=self.dtype), requires_grad=True),
                  
                  'log_w_kd': Parameter(log_w_kd.to(dtype=self.dtype), requires_grad=False),
                  'w_kd_sign': Parameter(w_kd_sign.to(dtype=self.dtype), requires_grad=False),
                  
                  'log_c_p': Parameter(log_c_p.to(dtype=self.dtype), requires_grad=False),
                  'c_p_sign': Parameter(c_p_sign.to(dtype=self.dtype), requires_grad=False)}
        self.register_params(params)
        
    def get_w_d(self):
        '''calculates the covariance for each hamming distance'''
        w_d = self.calc_w_d(self.w_kd_sign, self.log_w_kd, self.log_lambdas)
        return(w_d)
    
    def _nonkeops_forward_hamming_class(self, x1, x2, diag=False, **kwargs):
        w_d = self.get_w_d()
        hamming_dist = self.calc_hamming_distance(x1, x2).to(dtype=torch.long)
        kernel = w_d[0] * (hamming_dist == 0)
        for d in range(1, self.s):
            kernel += w_d[d] * (hamming_dist == d)
        return(kernel)
    
    def get_c_d(self):
        c_d = self.calc_c_d(self.c_p_sign, self.log_c_p, self.log_lambdas)
        return(c_d)

    def _nonkeops_forward_polynomial_d(self, x1, x2, diag=False, **kwargs):
        c_d = self.get_c_d()
        
        d = self.l - x1 @ x2.T
        k = torch.full_like(d, fill_value=c_d[0].item())
        d_power = 1
        for i in range(1, self.max_k + 1):
            d_power = d_power * d
            k += c_d[i] * d_power 
        return(k)

    def _nonkeops_forward(self, x1, x2, diag=False, **kwargs):
        if self.max_k < self.l:
            return(self._nonkeops_forward_polynomial_d(x1, x2, diag=diag, **kwargs))
        else:
            return(self._nonkeops_forward_hamming_class(x1, x2, diag=diag, **kwargs))
    
    def _covar_func(self, x1, x2, c_d, **kwargs):
        x1_ = LazyTensor(x1[..., :, None, :])
        x2_ = LazyTensor(x2[..., None, :, :])
        d = self.l - (x1_ * x2_).sum(-1)
        
        k = c_d[0]
        d_power = 1
        for i in range(1, self.max_k + 1):
            d_power = d_power * d
            k += c_d[i] * d_power 
        return(k)
        
    def _keops_forward(self, x1, x2, **kwargs):
        c_d = self.get_c_d()
        return(KernelLinearOperator(x1, x2, covar_func=self._covar_func,
                                    c_d=c_d, **kwargs))
    

class PairwiseKernel(VarianceComponentKernel):
    def __init__(self, n_alleles, seq_length, log_lambdas0=None,
                 log_var0=2., **kwargs):
        super().__init__(n_alleles, seq_length, log_lambdas0=log_lambdas0,
                         log_var0=log_var0, max_k=2, **kwargs)


class DeltaPKernel(VarianceComponentKernel):
    def __init__(self, n_alleles, seq_length, P, **kwargs):
        lambdas_prior = LambdasDeltaPrior(seq_length, n_alleles, P=P)
        super().__init__(n_alleles, seq_length, lambdas_prior=lambdas_prior,
                         **kwargs)


class RhoPiKernel(SequenceKernel):
    def __init__(self, n_alleles, seq_length,
                 logit_rho0=None, log_p0=None, log_var0=None, 
                 train_p=True, train_var=False,
                 common_rho=False, correlation=False,
                 **kwargs):
        super().__init__(n_alleles, seq_length, **kwargs)
        self.logit_rho0 = logit_rho0
        self.log_p0 = log_p0
        self.log_var0 = log_var0
        self.train_p = train_p
        self.train_var = train_var
        self.correlation = correlation
        self.common_rho = common_rho
        self.set_params()
    
    def get_log_p0(self):
        if self.log_p0 is None:
            log_p0 = -torch.ones((self.l, self.alpha), dtype=self.dtype) 
        else:
            log_p0 = self.log_p0
        return(log_p0)
    
    def get_logit_rho0(self):
        # Choose rho0 so that correlation at l/2 is 0.1
        shape = (1, 1) if self.common_rho else (self.l, 1)
        t = np.exp(-2 / self.l * np.log(10.))
        v = np.log((1 - t) / (self.alpha * t))
        logit_rho0 = torch.full(shape, v, dtype=self.dtype) if self.logit_rho0 is None else self.logit_rho0 
        return(logit_rho0)
    
    def get_log_var0(self, logit_rho0):
        if self.log_var0 is None:
            if self.correlation:
                rho = torch.exp(logit_rho0) / (1 + torch.exp(logit_rho0))
                log_var0 = torch.log(1 + (self.alpha - 1) * rho).sum()
            else:
                log_var0 = torch.tensor(0, dtype=self.dtype)    
        else:
            log_var0 = torch.tensor(self.log_var0, dtype=self.dtype)
        return(log_var0)
    
    def set_params(self):
        logit_rho0 = self.get_logit_rho0()
        log_p0 = self.get_log_p0()
        log_var0 = self.get_log_var0(logit_rho0=logit_rho0)
        params = {'logit_rho': Parameter(logit_rho0, requires_grad=True),
                  'log_p': Parameter(log_p0, requires_grad=self.train_p),
                  'log_var': Parameter(log_var0, requires_grad=self.train_var)}
        self.register_params(params)
        
    def get_log_eta(self):
        log_p = self.log_p - torch.logsumexp(self.log_p, axis=1).unsqueeze(1)
        log_eta = log1mexp(log_p) - log_p
        return(log_eta)
    
    def get_log_one_minus_rho(self):
        return(-torch.logaddexp(self.zeros_like(self.logit_rho), self.logit_rho))
    
    def get_factors(self):
        log1mrho = self.get_log_one_minus_rho()
        log_rho = self.logit_rho + log1mrho
        log_eta = self.get_log_eta()
        log_one_p_eta_rho = torch.logaddexp(self.zeros_like(log_rho), log_rho + log_eta)
        factors = log_one_p_eta_rho - log1mrho
        
        constant = log1mrho.sum()
        if self.common_rho:
            constant *= self.l
        constant += self.log_var
        return(constant, factors, log_one_p_eta_rho)
    
    def _nonkeops_forward(self, x1, x2, diag=False, **params):
        constant, factors, log_one_p_eta_rho = self.get_factors()
        factors = factors.reshape(1, self.t)
        log_one_p_eta_rho = log_one_p_eta_rho.reshape(self.t, 1)
        
        log_kernel = constant + x1 @ (x2 * factors).T
        if self.correlation:
            log_sd1 = 0.5 * (x1 @ log_one_p_eta_rho)
            log_sd2 = 0.5 * (x2 @ log_one_p_eta_rho).reshape((1, x2.shape[0]))
            log_kernel = log_kernel - log_sd1 - log_sd2
        
        kernel = torch.exp(log_kernel)
        return(kernel)
    
    def _covar_func(self, x1, x2, **kwargs):
        x1_ = LazyTensor(x1[..., :, None, :])
        x2_ = LazyTensor(x2[..., None, :, :])
        kernel = (x1_ * x2_).sum(-1).exp()
        return(kernel)
    
    def _keops_forward(self, x1, x2, **kwargs):
        constant, factors, log_one_p_eta_rho = self.get_factors()
        f = factors.reshape(1, self.t)
        log_one_p_eta_rho = log_one_p_eta_rho.reshape(1, self.t)
        c = torch.exp(constant)
        
        kernel = c * KernelLinearOperator(x1, x2 * f, covar_func=self._covar_func, **kwargs)
        if self.correlation:
            sd1_inv_D = DiagLinearOperator(torch.exp(-0.5 * (x1 * log_one_p_eta_rho).sum(1)))
            sd2_inv_D = DiagLinearOperator(torch.exp(-0.5 * (x2 * log_one_p_eta_rho).sum(1)))
            kernel = sd1_inv_D @ kernel @ sd2_inv_D

        return(kernel)
    

class RBFKernel(RhoPiKernel):
    is_stationary = True
    def __init__(self, n_alleles, seq_length, **kwargs):
        super().__init__(n_alleles, seq_length,
                         common_rho=True, train_p=False, train_var=False,
                         **kwargs)
    

class RhoKernel(RhoPiKernel):
    is_stationary = True
    def __init__(self, n_alleles, seq_length, **kwargs):
        super().__init__(n_alleles, seq_length,
                         common_rho=False, train_p=False, train_var=False,
                         **kwargs)

        
class ARDKernel(RhoPiKernel):
    is_stationary = True
    def __init__(self, n_alleles, seq_length, **kwargs):
        super().__init__(n_alleles, seq_length,
                         correlation=True, train_p=True, train_var=True,
                         **kwargs)
    
    
#################
# Skewed kernel #
#################

class _PiKernel(SequenceKernel):
    def set_p_prior(self, p_prior, dummy_allele=False):
        if p_prior is None:
            p_prior = AllelesProbPrior(seq_length=self.l, n_alleles=self.alpha,
                                       dummy_allele=dummy_allele)
        self.p_prior = p_prior
        self.p_prior.set(self)
    
    @property
    def beta(self):
        logp = -torch.exp(self.raw_logp)
        logp = self.p_prior.resize_logp(logp)
        norm_logp = self.p_prior.normalize_logp(logp)
        beta = self.p_prior.norm_logp_to_beta(norm_logp)
        return(beta)


class SkewedVCKernel(_LambdasKernel, _PiKernel):
    is_stationary = False
    def __init__(self, n_alleles, seq_length,
                 lambdas_prior=None, p_prior=None, q=None, **kwargs):
        super().__init__(n_alleles, seq_length, **kwargs)
        
        self.set_q(q)
        self.define_aux_variables()
        self.set_lambdas_prior(lambdas_prior)
        self.set_p_prior(p_prior, dummy_allele=True)
        
    def set_q(self, q=None):
        if q is None:
            q = (self.l - 1) / self.l
        self.q = q
        self.logq = np.log(q)

    def define_aux_variables(self):
        # Lambdas related parameters
        self.ks = np.arange(self.s)
        self.lambdas_p = np.exp(self.ks * self.logq)
        self.coeffs = Parameter(get_tensor(self.calc_polynomial_coeffs(), dtype=self.fdtype),
                                requires_grad=False)

        # q related parameters
        log_q_powers = self.ks * self.logq
        log_1mq_powers = np.append([-np.inf], np.log(1 - np.exp(log_q_powers[1:])))
        
        self.logq_powers = Parameter(get_tensor(log_q_powers, dtype=self.fdtype), requires_grad=False)
        self.log_1mq_powers = Parameter(get_tensor(log_1mq_powers, dtype=self.fdtype), requires_grad=False)
        
    @property
    def logp(self):
        logp = -torch.exp(self.raw_logp)
        logp = self.p_prior.normalize_logp(self.p_prior.resize_logp(logp))
        return(logp)

    @property
    def p(self):
        return(torch.exp(self.logp))
    
    def _forward(self, x1, x2, lambdas, norm_logp, diag=False):
        coeffs = self.coeffs.to(dtype=lambdas.dtype)
        c_ki = torch.matmul(coeffs, lambdas)
        coeff_signs = torch.ones_like(c_ki)
        coeff_signs[c_ki < 0] = -1
        log_c_ki = torch.log(torch.abs(c_ki))
        log_bi = log_c_ki + self.l * self.log_1mq_powers
        
        logp_flat = torch.flatten(norm_logp[:, :-1])
        beta = self.p_prior.norm_logp_to_beta(logp_flat)

        # Init first power
        M = torch.diag(logp_flat)
        if diag:
            kernel = coeff_signs[0] * torch.exp(log_c_ki[0]-(torch.matmul(x1, M) * x2).sum(1))
        else:
            kernel = coeff_signs[0] * torch.exp(log_c_ki[0]-self.inner_product(x1, x2, M))
            kernel *= torch.matmul(x1, x2.T) == self.l
        
        # Add the remaining powers        
        for power in range(1, self.s):
            weights = torch.log(1 + torch.exp(beta + self.logq_powers[power])) - self.log_1mq_powers[power]
            M = torch.diag(weights)
            m = self.inner_product(x1, x2, M, diag=diag)
            kernel += coeff_signs[power] * torch.exp(log_bi[power] + m)
        # print(x1.shape[0], x2.shape[0], kernel.shape)
        
        return(kernel)
    
    def forward(self, x1, x2, diag=False, **params):
        kernel = self._forward(x1, x2, lambdas=self.lambdas, norm_logp=self.logp,
                               diag=diag)
        return(kernel)
    
    def get_params(self):
        return({'lambdas': self.lambdas, 'beta': self.beta})
