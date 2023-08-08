import numpy as np
import torch as torch

from itertools import combinations
from scipy.special._basic import comb
from torch.nn import Parameter
from gpytorch.kernels.kernel import Kernel

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
    
    def get_params(self):
        return({'lambdas': self.lambdas})


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
        return({'lambdas': self.lambdas,
                'p': self.p})


class GeneralizedSiteProductKernel(SequenceKernel):
    def __init__(self, n_alleles, seq_length, p_prior, rho_prior,
                 dtype=torch.float32, **kwargs):
        super().__init__(n_alleles, seq_length, dtype=dtype, **kwargs)
        
        self.p_prior = p_prior
        self.rho_prior = rho_prior
        self.p_prior.set(self)
        self.rho_prior.set(self)

    @property
    def beta(self):
        logp = -torch.exp(self.raw_logp)
        logp = self.p_prior.resize_logp(logp)
        norm_logp = self.p_prior.normalize_logp(logp)
        beta = self.p_prior.norm_logp_to_beta(norm_logp)
        return(beta)
    
    @property
    def rho(self):
        return(self.rho_prior.get_rho(self))
    
    @property
    def rho_c(self):
        return(self.rho_prior.get_rho_c(self))
    
    def _forward(self, x1, x2, rho_c, rho, beta, diag=False):
        # TODO: make sure diag works here
        constant = torch.log(1 - rho).sum()
        rho = torch.stack([rho] * self.alpha, axis=1)
        eta = torch.exp(beta)
        log_factors = torch.flatten(torch.log(1 + rho * eta) - torch.log(1 - rho))
        M = torch.diag(log_factors)
        m = self.inner_product(x1, x2, M, diag=diag)
        kernel = torch.exp(m + constant) - 1 + rho_c
        return(kernel)
    
    def forward(self, x1, x2, diag=False, **params):
        return(self._forward(x1, x2, rho_c=self.rho_c, rho=self.rho, beta=self.beta, diag=diag))
    
    def get_params(self):
        return({'rho_c': self.rho_c, 
                'rho': self.rho,
                'beta': self.beta})


class DiploidKernel(SequenceKernel):
    def __init__(self, **kwargs):
        super().__init__(n_alleles=2, seq_length=0, **kwargs)
        self.define_kernel_params()
    
    def define_kernel_params(self):
#         constraints = {'raw_log_lda': LessThan(upper_bound=0.),
#                        'raw_log_eta': LessThan(upper_bound=0.)}
        params = {'raw_log_lda': Parameter(torch.zeros(1)),
                  'raw_log_eta': Parameter(torch.zeros(1)),
                  'raw_log_mu': Parameter(torch.zeros(1))}
        self.register_params(params=params) #constraints=constraints
    
    @property
    def mu(self):
        return(torch.exp(self.raw_log_mu))
    
    @property
    def lda(self):
#         return(torch.exp(self.raw_log_lda_constraint.transform(self.raw_log_lda)))
        return(torch.exp(self.raw_log_lda))

    @property
    def eta(self):
#         return(torch.exp(self.raw_log_eta_constraint.transform(self.raw_log_eta)))
        return(torch.exp(self.raw_log_eta))

    def dist(self, x1, x2):
        return(x1.matmul(x2.transpose(-2, -1)))
    
    def calc_distance_classes(self, x1, x2):
        l = x1.shape[1]
        s1 = self.dist(x1[:, :, 1], x2[:, :, 1])
        s2 = self.dist(x1[:, :, 0], x2[:, :, 0]) + self.dist(x1[:, :, 2], x2[:, :, 2])
        d2 = self.dist(x1[:, :, 0], x2[:, :, 2]) + self.dist(x1[:, :, 2], x2[:, :, 0])
        d1 = l - s1 - s2 - d2
        return(s2, d2, s1, d1)
    
    def _forward(self, x1, x2, mu, lda, eta):
        s2, d2, s1, d1 = self.calc_distance_classes(x1, x2)
        kernel = ((mu + 2 * lda + eta)**s2) * ((mu - 2 * lda + eta)**d2) * ((mu + eta)**s1) * ((mu - eta)**d1)
        return(kernel)

    def forward(self, x1, x2, **params):
        kernel = self._forward(x1, x2, self.mu, self.lda, self.eta)
        return(kernel)


class GeneralizedDiploidKernel(DiploidKernel):
    def __init__(self, seq_length, **kwargs):
        super().__init__(seq_length=seq_length, **kwargs)
        self.define_kernel_params()
    
    def define_kernel_params(self):
        super().define_kernel_params()
        params = {'raw_logit_p': Parameter(torch.zeros(1))}
        self.register_params(params=params)
    
    @property
    def odds(self):
        return(torch.exp(self.raw_logit_p))
    
    def _forward(self, x1, x2, mu, lda, eta, odds):
        s2, d2, s1, d1 = self.calc_distance_classes(x1, x2)
        kernel = (mu + (1 + odds) * lda + odds * eta)**s2 * (mu - (1 + odds) * lda + odds * eta)**d2 * (mu + odds * eta)**s1 * (mu - eta)**d1
        return(kernel)

    def forward(self, x1, x2, **params):
        kernel = self._forward(x1, x2, self.mu, self.lda, self.eta, self.odds)
        return(kernel)
