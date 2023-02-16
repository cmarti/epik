import numpy as np
import torch as torch

from itertools import combinations
from scipy.special._basic import comb
from torch.nn import Parameter
from gpytorch.kernels.kernel import Kernel
from gpytorch.constraints.constraints import LessThan

from epik.src.utils import get_tensor


class SequenceKernel(Kernel):
    is_stationary = True
    def __init__(self, n_alleles, seq_length, dtype=torch.float32, **kwargs):
        self.alpha = n_alleles
        self.l = seq_length
        self.s = self.l + 1
        self.t = self.l * self.alpha
        self.fdtype = dtype
        super().__init__(**kwargs)
    
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


class HaploidKernel(SequenceKernel):
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
    
    def calc_hamming_distance(self, x1, x2):
        return(self.l - self.inner_product(x1, x2))
    
    @property
    def lambdas(self):
        log_lambdas = self.lambdas_prior.theta_to_log_lambdas(self.raw_theta, kernel=self)
        log_lambdas = torch.cat((get_tensor([-10.], dtype=log_lambdas.dtype,
                                            device=log_lambdas.device), log_lambdas))
        lambdas = torch.exp(log_lambdas)
        return(lambdas)


class VCKernel(HaploidKernel):
    def __init__(self, n_alleles, seq_length, lambdas_prior,
                 **kwargs):
        super().__init__(n_alleles, seq_length, **kwargs)
        
        self.define_aux_variables()
        self.lambdas_prior = lambdas_prior
        self.lambdas_prior.set(self)

    def define_aux_variables(self):
        self.krawchouk_matrix = Parameter(self.calc_krawchouk_matrix(),
                                          requires_grad=False)
    
    def calc_w_kd(self, k, d):
        ss = 0
        for q in range(self.s):
            ss += (-1.)**q * (self.alpha-1.)**(k-q) * comb(d,q) * comb(self.l-d,k-q)
        return(ss)
    
    def calc_krawchouk_matrix(self):
        w_kd = np.zeros((self.s, self.s))
        for k in range(self.s):
            for d in range(self.s):
                w_kd[d, k] = self.calc_w_kd(k, d)
        return(get_tensor(w_kd))
    
    def _forward(self, x1, x2, lambdas, diag=False):
        w_d = torch.matmul(self.krawchouk_matrix, lambdas)
        hamming_dist = self.calc_hamming_distance(x1, x2).to(dtype=torch.long)
        kernel = w_d[0] * (hamming_dist == 0)
        for d in range(1, self.s):
            kernel += w_d[d] * (hamming_dist == d)
        return(kernel)
    
    def forward(self, x1, x2, diag=False, **params):
        kernel = self._forward(x1, x2, lambdas=self.lambdas, diag=diag)
        return(kernel)


class SkewedVCKernel(HaploidKernel):
    def __init__(self, n_alleles, seq_length, lambdas_prior, p_prior, q=None,
                 dtype=torch.float32, **kwargs):
        super().__init__(n_alleles, seq_length, dtype=dtype, **kwargs)
        
        self.set_q(q)
        self.define_aux_variables()
        self.lambdas_prior = lambdas_prior
        self.lambdas_prior.set(self)
        self.p_prior = p_prior
        self.p_prior.set(self)
        
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
        return(kernel)
    
    def forward(self, x1, x2, diag=False, **params):
        kernel = self._forward(x1, x2, lambdas=self.lambdas, norm_logp=self.logp,
                               diag=diag)
        return(kernel)


class SiteProductKernel(HaploidKernel):
    def __init__(self, n_alleles, seq_length, p_prior,
                 dtype=torch.float32, **kwargs):
        super().__init__(n_alleles, seq_length, dtype=dtype, **kwargs)
        
        self.define_params()
        self.p_prior = p_prior
        self.p_prior.set(self)

    def define_params(self):
        params = {'raw_theta': Parameter(torch.zeros(2), requires_grad=True)}
        self.register_params(params=params, constraints={})
        
    @property
    def beta(self):
        logp = -torch.exp(self.raw_logp)
        logp = self.p_prior.resize_logp(logp)
        norm_logp = self.p_prior.normalize_logp(logp)
        beta = self.p_prior.norm_logp_to_beta(norm_logp)
        return(beta)
    
    @property
    def theta(self):
        return(self.raw_theta)

    def _forward(self, x1, x2, theta, beta, diag=False):
        # TODO: make sure diag works here
        log_factors = torch.flatten(torch.log(torch.exp(theta[0]) + torch.exp(theta[1] + beta[:, :-1])))
        M = torch.diag(log_factors)
        m = self.inner_product(x1, x2, M, diag=diag)
        
        distance = self.l - self.inner_product(x1, x2, diag=diag)
        kernel = torch.exp(m) * (torch.exp(theta[0]) - torch.exp(theta[1])) ** distance # 1-et can be negative: avoid log
        return(kernel)
    
    def forward(self, x1, x2, diag=False, **params):
        return(self._forward(x1, x2, theta=self.theta, beta=self.beta, diag=diag))


class DiploidKernel(SequenceKernel):
    def __init__(self, seq_length, **kwargs):
        super().__init__(n_alleles=2, seq_length=seq_length, **kwargs)
        self.define_kernel_params()
    
    def define_kernel_params(self):
        constraints = {'raw_log_lda': LessThan(upper_bound=0.),
                       'raw_log_eta': LessThan(upper_bound=0.)}
        params = {'raw_log_lda': Parameter(torch.zeros(*self.batch_shape, 1, 1)),
                  'raw_log_eta': Parameter(torch.zeros(*self.batch_shape, 1, 1))}
        self.register_params(params=params, constraints=constraints)
    
    @property
    def log_lda(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_log_lda_constraint.transform(self.raw_log_lda)

    @property
    def log_eta(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_log_eta_constraint.transform(self.raw_log_eta)

    @log_lda.setter
    def log_lda(self, value):
        return self._set_log_lda(value)

    @log_eta.setter
    def log_eta(self, value):
        return self._set_log_eta(value)
  
    def _forward(self, log_lda, log_eta, S1, S2, D2):
        L = self.seq_length
        lda = torch.exp(log_lda)
        eta = torch.exp(log_eta)
        kernel = (((1 + lda + eta)**(S2 - L/2)) *((1 - lda + eta)**D2) *((1 + eta)**(S1 - L/2)) * (1 - eta)**((L - S1 - S2 - D2)))
        return(kernel)
    
    def dist(self, x1, x2):
        res = x1.matmul(x2.transpose(-2, -1))
        return(res)   

    def forward(self, geno1, geno2, **params):
        geno1_ht = 1.*(geno1 == 1.)
        geno2_ht = 1.*(geno2 == 1.)        
        geno1_h0 = 1.*(geno1 == 0.)
        geno1_h1 = 1.*(geno1 == 2.)
        geno2_h0 = 1.*(geno2 == 0.)
        geno2_h1 = 1.*(geno2 == 2.)

        S1 = self.dist(geno1_ht, geno2_ht)
        S2 = self.dist(geno1_h0, geno2_h0) + self.dist(geno1_h1, geno2_h1)
        D2 = self.dist(geno1_h0, geno2_h1) + self.dist(geno1_h1, geno2_h0)

        kernel = self._forward(self.log_lda, self.log_eta, S1, S2, D2)
        return(kernel)
