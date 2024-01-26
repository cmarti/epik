import numpy as np
import torch as torch

from itertools import combinations
from scipy.special._basic import comb
from torch.nn import Parameter
from pykeops.torch import LazyTensor
from linear_operator.operators import KernelLinearOperator

from epik.src.utils import get_tensor, log1mexp
from epik.src.kernel.base import SequenceKernel
from epik.src.priors import (LambdasFlatPrior, LambdasDeltaPrior, RhosPrior,
                             AllelesProbPrior)


class AdditiveKernel(SequenceKernel):
    def __init__(self, n_alleles, seq_length, log_lambdas0=None, **kwargs):
        super().__init__(n_alleles, seq_length, **kwargs)
        self.log_lambdas0 = log_lambdas0
        self.set_params()
    
    def get_matrix(self):
        m = torch.tensor([[1, -self.l], [0., self.alpha]])
        return(m)  
    
    def set_params(self):
        log_lambdas0 = torch.tensor([0, 0.]) if self.log_lambdas0 is None else self.log_lambdas0
        params = {'log_lambdas': Parameter(log_lambdas0, requires_grad=True),
                  'm': Parameter(self.get_matrix(), requires_grad=False)}
        self.register_params(params)
    
    def lambdas_to_coeffs(self, lambdas):
        coeffs = self.m @ lambdas
        return(coeffs)

    def get_coeffs(self):
        return(self.lambdas_to_coeffs(torch.exp(self.log_lambdas)))
    
    def forward(self, x1, x2, diag=False, **kwargs):
        coeffs = self.get_coeffs()
        return(coeffs[0] + coeffs[1] * (x1 @ x2.T))


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
    
    @property
    def lambdas(self):
        log_lambdas = self.lambdas_prior.theta_to_log_lambdas(self.raw_theta, kernel=self)
        lambdas = torch.exp(log_lambdas)
        return(lambdas)
    
    def set_lambdas_prior(self, lambdas_prior):
        if lambdas_prior is None:
            lambdas_prior = LambdasFlatPrior(seq_length=self.l)
        self.lambdas_prior = lambdas_prior
        self.lambdas_prior.set(self)


class VarianceComponentKernel(_LambdasKernel):
    is_stationary = True
    def __init__(self, n_alleles, seq_length, lambdas_prior=None,
                 **kwargs):
        super().__init__(n_alleles, seq_length, **kwargs)
        
        self.define_aux_variables()
        self.set_lambdas_prior(lambdas_prior)

    def define_aux_variables(self):
        self.krawchouk_matrix = Parameter(self.calc_krawchouk_matrix(),
                                          requires_grad=False)
    
    def calc_w_kd(self, k, d):
        ss = 0
        for q in range(self.s):
            ss += (-1.)**q * (self.alpha-1.)**(k-q) * comb(d,q) * comb(self.l-d,k-q)
        return(ss / self.n)
    
    def calc_krawchouk_matrix(self):
        w_kd = np.zeros((self.s, self.s))
        for k in range(self.s):
            for d in range(self.s):
                w_kd[d, k] = self.calc_w_kd(k, d)
        return(get_tensor(w_kd))
    
    def _forward(self, x1, x2, lambdas, diag=False):
        w_d = self.krawchouk_matrix @ lambdas
        hamming_dist = self.calc_hamming_distance(x1, x2).to(dtype=torch.long)
        kernel = w_d[0] * (hamming_dist == 0)
        for d in range(1, self.s):
            kernel += w_d[d] * (hamming_dist == d)
        return(kernel)
    
    def forward(self, x1, x2, diag=False, **params):
        kernel = self._forward(x1, x2, lambdas=self.lambdas, diag=diag)
        return(kernel)
    
    def get_params(self):
        return({'lambdas': self.lambdas})


class DeltaPKernel(VarianceComponentKernel):
    def __init__(self, n_alleles, seq_length, P, **kwargs):
        lambdas_prior = LambdasDeltaPrior(seq_length, n_alleles, P=P)
        super().__init__(n_alleles, seq_length, lambdas_prior=lambdas_prior,
                         **kwargs)


class RhoPiKernel(SequenceKernel):
    def __init__(self, n_alleles, seq_length,
                 logit_rho0=None, log_p0=None,
                 train_p=True, common_rho=False, correlation=False,
                 **kwargs):
        super().__init__(n_alleles, seq_length, **kwargs)
        self.logit_rho0 = logit_rho0
        self.log_p0 = log_p0
        self.train_p = train_p
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
        t = np.exp(-2/self.l * np.log(10.))
        v = np.log((1 - t) / (self.alpha * t))
        logit_rho0 = torch.full(shape, v, dtype=self.dtype) if self.logit_rho0 is None else self.logit_rho0 
        return(logit_rho0)
    
    def set_params(self):
        logit_rho0 = self.get_logit_rho0()
        log_p0 = self.get_log_p0()
        params = {'logit_rho': Parameter(logit_rho0, requires_grad=True),
                  'log_p': Parameter(log_p0, requires_grad=self.train_p)}
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
        
        return(constant, factors, log_one_p_eta_rho)
    
    def _nonkeops_forward(self, x1, x2, diag=False, **params):
        constant, factors, log_one_p_eta_rho = self.get_factors()
        factors = factors.reshape(1, self.t)
        log_one_p_eta_rho = log_one_p_eta_rho.reshape(self.t, 1)
        
        if self.correlation:
            log_sd1 = 0.5 * (x1 @ log_one_p_eta_rho)
            log_sd2 = 0.5 * (x2 @ log_one_p_eta_rho).reshape((1, x2.shape[0]))
            kernel = torch.exp(constant + x1 @ (x2 * factors).T - log_sd1 - log_sd2)
        else:
            kernel = torch.exp(constant + x1 @ (x2 * factors).T)
        # print(kernel)
        return(kernel)
    
    def _covar_func(self, x1, x2, **kwargs):
        x1_ = LazyTensor(x1[..., :, None, :])
        x2_ = LazyTensor(x2[..., None, :, :])
        kernel = ((x1_ * x2_).sum(-1)).exp()
        return(kernel)
    
    def _keops_forward(self, x1, x2, **kwargs):
        constant, factors, log_one_p_eta_rho = self.get_factors()
        c = torch.exp(constant)
        f = factors.T.reshape(1, self.t)
        log_one_p_eta_rho = log_one_p_eta_rho.T.reshape(1, self.t)
        
        kernel = c * KernelLinearOperator(x1, x2 * f, covar_func=self._covar_func, **kwargs)
        if self.correlation:
            sd1 = torch.exp(0.5 * (x1 * log_one_p_eta_rho).sum(1)).reshape((x1.shape[0], 1))
            sd2 = torch.exp(0.5 * (x2 * log_one_p_eta_rho).sum(1)).reshape((x2.shape[0], 1))
            kernel = kernel / (sd1 * sd2.T)
        return(kernel)
    
    def get_params(self):
        return({'logit_rho': self.logit_rho,
                'log_p': self.log_p})
    

class RhoKernel(RhoPiKernel):
    is_stationary = True
    def __init__(self, n_alleles, seq_length, logit_rho0=None,
                 common_rho=False, **kwargs):
        super().__init__(n_alleles, seq_length, logit_rho0=logit_rho0, common_rho=common_rho,
                         train_p=False, log_p0=None, **kwargs)
    
    
class RBFKernel(RhoKernel):
    is_stationary = True
    def __init__(self, n_alleles, seq_length, logit_rho0=None, **kwargs):
        super().__init__(n_alleles, seq_length, logit_rho0=logit_rho0, common_rho=True, **kwargs)
        
        
class ARDKernel(RhoPiKernel):
    is_stationary = True
    def __init__(self, n_alleles, seq_length, **kwargs):
        super().__init__(n_alleles, seq_length, correlation=True, train_p=True, **kwargs)
    
    
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
