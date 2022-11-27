import numpy as np
import torch

from itertools import combinations

from torch.nn import Parameter
from gpytorch.kernels.kernel import Kernel
from gpytorch.priors.torch_priors import NormalPrior

from epik.src.utils import to_device, get_tensor
from gpytorch.constraints.constraints import LessThan
from scipy.special._basic import comb


class SequenceKernel(Kernel):
    is_stationary = True
    def __init__(self, n_alleles, seq_length, q=None, **kwargs):
        self.alpha = n_alleles
        self.l = seq_length
        self.s = self.l + 1
        self.t = self.l * self.alpha
        if q is None:
            q = (self.l - 1) / self.l
        self.q = q

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
    
    def get_log_lda_to_theta_matrix(self):
        matrix = torch.zeros((self.l, self.l))
        matrix[0, 0] = 1
        matrix[1, 0] = -1
        matrix[1, 1] = 1
        for i in range(2, self.l):
            matrix[i, i-2] = -1
            matrix[i, i-1] = 2
            matrix[i, i] = -1
        return(matrix) 
    
    def get_theta_to_log_lda_matrix(self):
        return(torch.inverse(self.get_log_lda_to_theta_matrix()))


class VCKernel(SequenceKernel):
    def __init__(self, n_alleles, seq_length, tau=0.2,
                 train_lambdas=True, starting_log_lambdas=None,
                 **kwargs):
        super().__init__(n_alleles, seq_length, **kwargs)
        
        self.train_lambdas = train_lambdas
        self.starting_log_lambdas = starting_log_lambdas
        
        self.define_aux_variables()
        self.define_kernel_params()
        self.define_priors(tau)

    def define_aux_variables(self):
        self.log_lda_to_theta_m = Parameter(self.get_log_lda_to_theta_matrix(),
                                            requires_grad=False)
        self.theta_to_log_lda_m = Parameter(self.get_theta_to_log_lda_matrix(),
                                            requires_grad=False)
        self.krawchouk_matrix = self.calc_krawchouk_matrix()
    
    def calc_w_kd(self, k, d):
        ss = 0
        for q in range(self.s):
            ss += (-1)**q * (self.alpha-1)**(k-q) * comb(d,q) * comb(self.l-d,k-q)
        return(ss)
    
    def calc_krawchouk_matrix(self):
        w_kd = np.zeros((self.s, self.s))
        for k in range(self.s):
            for d in range(self.s):
                w_kd[d, k] = self.calc_w_kd(k, d)
        return(get_tensor(w_kd))
    
    def define_kernel_params(self):
        if self.starting_log_lambdas is None:
            raw_theta0 = torch.zeros(self.l)
            raw_theta0[0:2] = -1
        else:
            raw_theta0 = torch.matmul(self.log_lda_to_theta_m,
                                      get_tensor(self.starting_log_lambdas))
        params = {'raw_theta': Parameter(raw_theta0, requires_grad=self.train_lambdas)}
        self.register_params(params=params, constraints={})
    
    def define_priors(self, tau):
        self.register_prior("raw_theta_prior", NormalPrior(0, tau),
                            lambda module: module.raw_theta[2:])
        self.register_prior("raw_theta_prior1", NormalPrior(-1, 1),
                            lambda module: module.raw_theta[1])
        self.register_prior("raw_theta_prior0", NormalPrior(-1, 1),
                            lambda module: module.raw_theta[0])

    @property
    def log_lda(self):
        return(torch.matmul(self.theta_to_log_lda_m, self.raw_theta))
    
    @property
    def lambdas(self):
        lambdas = to_device(torch.zeros(self.s), 
                            output_device=self.log_lda.get_device())
        lambdas[1:] = torch.exp(self.log_lda)
        return(lambdas)
    
    @property
    def log_p(self):
        return(self.raw_log_p)

    @log_lda.setter
    def log_lda(self, value):
        return self._set_log_lda(value)

    def _forward(self, x1, x2, lambdas, diag=False):
        w_d = torch.matmul(self.krawchouk_matrix, lambdas)
        hamming_dist = (self.l - torch.matmul(x1, x2.T)).to(dtype=torch.long)
        kernel = w_d[hamming_dist]
        return(kernel)
    
    def forward(self, x1, x2, diag=False, **params):
        kernel = self._forward(x1, x2, lambdas=self.lambdas, diag=diag)
        return(kernel)


class SkewedVCKernel(SequenceKernel):
    def __init__(self, n_alleles, seq_length, q=None, tau=0.2,
                 train_p=True, train_lambdas=True,
                 starting_p=None, starting_log_lambdas=None,
                 **kwargs):
        super().__init__(n_alleles, seq_length, q=q, **kwargs)
        
        self.train_p = train_p
        self.train_lambdas = train_lambdas
        self.starting_p = starting_p 
        self.starting_log_lambdas = starting_log_lambdas
        
        self.define_aux_variables()
        self.define_kernel_params()
        self.define_priors(tau)

    def calc_eigenvalues(self):
        lambdas = self.q ** torch.arange(self.s)
        return(lambdas)

    def define_aux_variables(self):
        self.q_powers = torch.pow(self.q, torch.arange(self.s))
        self.coeffs = Parameter(self.calc_polynomial_coeffs(), requires_grad=False)
        self.log_lda_to_theta_m = Parameter(self.get_log_lda_to_theta_matrix(),
                                            requires_grad=False)
        self.theta_to_log_lda_m = Parameter(self.get_theta_to_log_lda_matrix(),
                                            requires_grad=False)
        
        lsf = self.l * torch.log(1 - self.q_powers)
        self.log_scaling_factors = Parameter(lsf, requires_grad=False)
        
        log_odds = torch.log(self.q_powers) - torch.log(1 - self.q_powers)
        self.log_odds = Parameter(log_odds, requires_grad=False)
    
    def define_kernel_params(self):
        constraints = {}
        
        if self.starting_p is None:
            starting_logp = np.zeros((self.l, self.alpha + 1))
            starting_logp[:, -1] = -10
            starting_logp = get_tensor(starting_logp)
        else:
            starting_logp = get_tensor(torch.log(self.starting_p))
            starting_logp[:, -1] = -10.
        params = {'raw_log_p': Parameter(starting_logp, requires_grad=self.train_p)}
        
        if self.starting_log_lambdas is None:
            raw_theta0 = torch.zeros(self.l)
            raw_theta0[0:2] = -1
        else:
            raw_theta0 = torch.matmul(self.log_lda_to_theta_m,
                                      get_tensor(self.starting_log_lambdas))
        params['raw_theta'] = Parameter(raw_theta0, requires_grad=self.train_lambdas)
        
        self.register_params(params=params, constraints=constraints)
    
    def define_priors(self, tau):
        self.register_prior("raw_theta_prior", NormalPrior(0, tau),
                            lambda module: module.raw_theta[2:])
        self.register_prior("raw_theta_prior1", NormalPrior(-1, 1),
                            lambda module: module.raw_theta[1])
        self.register_prior("raw_theta_prior0", NormalPrior(-1, 1),
                            lambda module: module.raw_theta[0])

    @property
    def log_lda(self):
        return(torch.matmul(self.theta_to_log_lda_m, self.raw_theta))
    
    @property
    def lambdas(self):
        lambdas = to_device(torch.zeros(self.s), 
                            output_device=self.log_lda.get_device())
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
    
    def _forward(self, x1, x2, lambdas, log_p, diag=False):
        log_p = self.normalize_log_p(log_p)
        log_p_flat = torch.flatten(log_p[:, :-1])
        c_ki = torch.matmul(self.coeffs, lambdas)
#         print(c_ki, 'c_ki')

        # Init first power
        M = torch.diag(log_p_flat)
        if diag:
            kernel = c_ki[0] * torch.exp(-(torch.matmul(x1, M) * x2).sum(1))
        else:
            kernel = c_ki[0] * torch.exp(-torch.matmul(torch.matmul(x1, M), x2.T))
            kernel[torch.matmul(x1, x2.T) < self.l] = 0
        
        # Add the remaining powers        
        for power in range(1, self.s):
            log_factors = torch.stack([self.log_odds[power] - log_p_flat,
                                       torch.zeros_like(log_p_flat)], 1)
            log_factors = torch.logsumexp(log_factors, 1)
            M = torch.diag(log_factors)
            if diag:
                kernel += c_ki[power] * torch.exp(self.log_scaling_factors[power] + (torch.matmul(x1, M) * x2).sum(1))
            else:
                kernel += c_ki[power] * torch.exp(self.log_scaling_factors[power] + torch.matmul(torch.matmul(x1, M), x2.T))
        return(kernel)
    
    def forward(self, x1, x2, diag=False, **params):
        kernel = self._forward(x1, x2, lambdas=self.lambdas, log_p=self.log_p,
                               diag=diag)
        return(kernel)
    

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
