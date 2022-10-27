import numpy as np
import gpytorch
import torch

from itertools import combinations

from torch.nn import Parameter
from gpytorch.constraints import LessThan


class SequenceKernel(gpytorch.kernels.kernel.Kernel):
    is_stationary = True
    def __init__(self, n_alleles, seq_length, **kwargs):
        self.alpha = n_alleles
        self.l = seq_length
        self.s = self.l + 1
        self.t = self.l * self.alpha
        self.q = (self.l - 1) / self.l
        
        super().__init__(**kwargs)
    
    def register_params(self, params={}, constraints={}):
        self.params = params
        for param_name, param in params.items():
            self.register_parameter(name=param_name, parameter=param)
            
        for param_name, constraint in constraints.items():
            self.register_constraint(param_name, constraint)
    
    def calc_polynomial_coeffs(self):
        lambdas = self.calc_eigenvalues()
        
        B = torch.zeros((self.s, self.s))
        idx = torch.arange(self.s)
        for k in idx:
            k_idx = idx != k
            k_lambdas = lambdas[k_idx]
            norm_factor = 1 / torch.prod(k_lambdas - lambdas[k])
    
            for power in idx:
                p = torch.tensor([np.product(v)
                                  for v in combinations(k_lambdas, self.l - power)]).sum()
                B[power, k] = norm_factor * (-1) ** (power) * p
        return(B)


class SkewedVCKernel(SequenceKernel):
    def __init__(self, n_alleles, seq_length, train_p=True,
                 force_exp_decay=False, 
                 log_lda_prior=None, log_p_prior=None,
                 **kwargs):
        
        super().__init__(n_alleles, seq_length, **kwargs)
        
        self.train_p = train_p
        self.force_exp_decay = force_exp_decay
        self.define_aux_variables()
        self.define_kernel_params()
    
    def calc_eigenvalues(self):
        lambdas = self.q ** torch.arange(self.s)
        return(lambdas)

    def define_aux_variables(self):
        self.q_powers = torch.pow(self.q, torch.arange(self.s))
        self.coeffs = Parameter(self.calc_polynomial_coeffs(), requires_grad=False)
        
        lsf = self.l * torch.log(1 - self.q_powers)
        self.log_scaling_factors = Parameter(lsf, requires_grad=False)
        
        log_odds = torch.log(self.q_powers) - torch.log(1 - self.q_powers)
        self.log_odds = Parameter(log_odds, requires_grad=False)
    
    def define_kernel_params(self):
        constraints = {}
        params = {'raw_log_p': Parameter(torch.zeros(*self.batch_shape, self.l, self.alpha),
                                         requires_grad=self.train_p)}

        if self.force_exp_decay:
            params['raw_log_lda_alpha'] = Parameter(torch.zeros(*self.batch_shape, 1)+1)
            params['raw_log_lda_beta'] = Parameter(torch.zeros(*self.batch_shape, 1)-10)
            constraints['raw_log_lda_beta'] = LessThan(upper_bound=0.)
        else:
            params['raw_log_lda'] = Parameter(-2.0*torch.arange(self.l))
        
        self.register_params(params=params, constraints=constraints)

    @property
    def log_lda_alpha(self):
        return(self.raw_log_lda_alpha)
    
    @property
    def log_lda_beta(self):
        return(self.raw_log_lda_beta_constraint.transform(self.raw_log_lda_beta))

    @property
    def log_lda(self):
        if self.force_exp_decay:
            return(self.log_lda_alpha + self.log_lda_beta * torch.arange(self.l))
        return(self.raw_log_lda)
    
    @property
    def lambdas(self):
        lambdas = torch.zeros(self.s)
        lambdas[1:] = torch.exp(self.log_lda)
        return(lambdas)
    
    @property
    def log_p(self):
        return(self.raw_log_p)

    def normalize_log_p(self, log_p):
        return(log_p - torch.logsumexp(log_p, 1).unsqueeze(1))
    
    @property
    def p(self):
        return(torch.exp(self.normalize_log_p(self.log_p)))

    @log_lda.setter
    def log_lda(self, value):
        return self._set_log_lda(value)

    @log_p.setter
    def log_p(self, value):
        return self._set_log_p(value)
    
    def _forward(self, x1, x2, lambdas, log_p):
        log_p = self.normalize_log_p(log_p)
        log_p_flat = torch.flatten(log_p)
        c_ki = torch.matmul(self.coeffs, lambdas)
        common = torch.matmul(x1, x2.T)

        # Init first power
        M = torch.diag(log_p_flat)
        kernel = c_ki[0] * torch.exp(-torch.matmul(torch.matmul(x1, M), x2.T))
        kernel[common < self.l] = 0
        
        # Add the remaining powers        
        for power in range(1, self.s):
            log_factors = torch.stack([self.log_odds[power] - log_p_flat,
                                       torch.zeros_like(log_p_flat)], 1)
            log_factors = torch.logsumexp(log_factors, 1)
            M = torch.diag(log_factors)
            kernel += c_ki[power] * torch.exp(self.log_scaling_factors[power] + torch.matmul(torch.matmul(x1, M), x2.T))
        return(kernel)

    def forward(self, x1, x2, diag=False, **params):
        kernel = self._forward(x1, x2, lambdas=self.lambdas, log_p=self.log_p)
        return(kernel)
