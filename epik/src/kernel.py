import numpy as np
import torch as torch

from itertools import combinations

from torch.nn import Parameter
from gpytorch.kernels.kernel import Kernel
from gpytorch.priors.torch_priors import NormalPrior

from epik.src.utils import to_device, get_tensor
from gpytorch.constraints.constraints import LessThan, GreaterThan
from scipy.special._basic import comb


class SequenceKernel(Kernel):
    is_stationary = True
    def __init__(self, n_alleles, seq_length, q=None, dtype=torch.float32, **kwargs):
        self.alpha = n_alleles
        self.l = seq_length
        self.s = self.l + 1
        self.t = self.l * self.alpha
        if q is None:
            q = (self.l - 1) / self.l
        self.q = q
        self.logq = np.log(q)
        self.fdtype = dtype

        super().__init__(**kwargs)
    
    def register_params(self, params={}, constraints={}):
        self.params = params
        for param_name, param in params.items():
            self.register_parameter(name=param_name, parameter=param)
            
        for param_name, constraint in constraints.items():
            self.register_constraint(param_name, constraint)
    
    def calc_polynomial_coeffs(self):
        lambdas = self.calc_eigenvalues()
        
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
    
    def inner_product(self, x1, x2, metric=None, diag=False):
        if metric is None:
            metric = torch.eye(x2.shape[1], dtype=x2.dtype, device=x2.device)
            
        if diag:
            min_size = min(x1.shape[0], x2.shape[0])
            return((torch.matmul(x1[:min_size, :], metric) * x2[:min_size, :]).sum(1))
        else:
            return(torch.matmul(x1, torch.matmul(metric, x2.T)))

    def calc_hamming_distance(self, x1, x2):
        return(self.l - self.inner_product(x1, x2))


class ExponentialKernel(SequenceKernel):
    def __init__(self, n_alleles, seq_length,
                 train_p=True, starting_p=None,
                 starting_lengthscale=1.,
                 **kwargs):
        super().__init__(n_alleles, seq_length, **kwargs)
        
        self.train_p = train_p
        self.starting_p = starting_p
        self.starting_lengthscale = starting_lengthscale 
        self.logn = seq_length * np.log(n_alleles)
        
        self.define_kernel_params()

    def define_kernel_params(self):
        if self.starting_p is None:
            starting_logp = np.zeros((self.l, self.alpha + 1))
            starting_logp[:, -1] = -10
            starting_logp = get_tensor(starting_logp)
        else:
            starting_logp = get_tensor(torch.log(self.starting_p))
            starting_logp[:, -1] = -10.
        params = {'raw_log_p': Parameter(starting_logp, requires_grad=self.train_p),
                  'raw_lengthscale': Parameter(get_tensor([self.starting_lengthscale]), requires_grad=True)}
        constraints = {'raw_lengthscale': GreaterThan(0.)}
        
        self.register_params(params=params, constraints=constraints)
    
    @property
    def log_p(self):
        return(self.raw_log_p)
    
    @property
    def lengthscale(self):
        return(self.raw_lengthscale)

    def normalize_log_p(self, log_p):
        return(log_p - torch.logsumexp(log_p, 1).unsqueeze(1))
    
    @property
    def p(self):
        return(torch.exp(self.normalize_log_p(self.log_p)))

    @log_p.setter
    def log_p(self, value):
        return self._set_log_p(value)
    
    @lengthscale.setter
    def lengthscale(self, value):
        return self._set_lengthscale(value)
    
    def _forward(self, x1, x2, lengthscale, log_p, diag=False):
        log_p = self.normalize_log_p(log_p)
        log_p_flat = torch.flatten(log_p[:, :-1])
        log_factors = torch.stack([-log_p_flat, torch.zeros_like(log_p_flat)], 1)
        log_factors = torch.logsumexp(log_factors, 1)
        M = torch.diag(log_factors)
        similarity = self.inner_product(x1, x2, M, diag=diag)
        m = self.l * np.log(1 + self.alpha)
        kernel = torch.exp(lengthscale/np.log2(self.alpha+1) * (similarity - m))
        return(kernel)
    
    def forward(self, x1, x2, diag=False, **params):
        kernel = self._forward(x1, x2, lengthscale=self.lengthscale, log_p=self.log_p,
                               diag=diag)
        return(kernel)


class VCKernel(SequenceKernel):
    def __init__(self, n_alleles, seq_length, tau=0.2,
                 train_lambdas=True, log_lambdas0=None,
                 **kwargs):
        super().__init__(n_alleles, seq_length, **kwargs)
        
        self.train_lambdas = train_lambdas
        self.log_lambdas0 = log_lambdas0
        
        self.define_aux_variables()
        self.define_kernel_params()
        self.define_priors(tau)

    def define_aux_variables(self):
        self.log_lda_to_theta_m = Parameter(self.get_log_lda_to_theta_matrix(),
                                            requires_grad=False)
        self.theta_to_log_lda_m = Parameter(self.get_theta_to_log_lda_matrix(),
                                            requires_grad=False)
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
    
    def define_kernel_params(self):
        if self.log_lambdas0 is None:
            raw_theta0 = torch.zeros(self.l)
            raw_theta0[0:2] = -1
        else:
            raw_theta0 = torch.matmul(self.log_lda_to_theta_m,
                                      get_tensor(self.log_lambdas0))
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
        hamming_dist = self.calc_hamming_distance(x1, x2).to(dtype=torch.long)
        kernel = w_d[hamming_dist]
        return(kernel)
    
    def forward(self, x1, x2, diag=False, **params):
        kernel = self._forward(x1, x2, lambdas=self.lambdas, diag=diag)
        return(kernel)


class SkewedVCKernel(SequenceKernel):
    def __init__(self, n_alleles, seq_length, q=None, tau=0.2,
                 train_p=True, train_lambdas=True,
                 starting_p=None, log_lambdas0=None,
                 dtype=torch.float32, lambdas_prior='monotonic_decay',
                 **kwargs):
        super().__init__(n_alleles, seq_length, q=q, dtype=dtype, **kwargs)
        
        self.define_aux_variables()
        self.define_lambdas(tau=tau, lambdas_prior=lambdas_prior,
                            train_lambdas=train_lambdas,
                            log_lambdas0=log_lambdas0)
        self.define_ps(starting_p=starting_p, train_p=train_p) 

    def calc_eigenvalues(self):
        k = np.arange(self.s)
        lambdas = np.exp(k * self.logq) 
        return(lambdas)

    def define_aux_variables(self):
        # Lambdas related parameters
        self.coeffs = Parameter(get_tensor(self.calc_polynomial_coeffs(), dtype=self.fdtype),
                                requires_grad=False)
        self.log_lda_to_theta_m = Parameter(self.get_log_lda_to_theta_matrix().to(self.fdtype),
                                            requires_grad=False)
        self.theta_to_log_lda_m = Parameter(self.get_theta_to_log_lda_matrix().to(self.fdtype),
                                            requires_grad=False)

        # p related parameters
        ks = np.arange(self.s)
        log_q_powers = self.logq * ks
        log_1mq_powers = np.append([-np.inf], np.log(1 - np.exp(log_q_powers[1:])))
        lsf = self.l * log_1mq_powers
        self.log_scaling_factors = Parameter(get_tensor(lsf, dtype=self.fdtype), requires_grad=False)
        
        log_odds = log_q_powers - log_1mq_powers
        self.log_odds = Parameter(get_tensor(log_odds, dtype=self.fdtype), requires_grad=False)
    
    def define_ps(self, train_p, starting_p=None):
        if starting_p is None:
            starting_logp = np.zeros((self.l, self.alpha + 1))
            starting_logp[:, -1] = -10
            starting_logp = get_tensor(starting_logp, dtype=self.fdtype)
        else:
            starting_logp = get_tensor(torch.log(starting_p), dtype=self.fdtype)
            starting_logp[:, -1] = -10.
            
        params = {'raw_log_p': Parameter(starting_logp, requires_grad=train_p)}
        self.register_params(params=params, constraints={})
        
    def define_lambdas_2nd_order(self):
        # Set parametrization based on the 2nd order differences
        if self.log_lambdas0 is None:
            raw_theta0 = torch.zeros(self.l).to(self.fdtype)
            raw_theta0[0:2] = -1
        else:
            raw_theta0 = torch.matmul(self.log_lda_to_theta_m,
                                      get_tensor(self.log_lambdas0, dtype=self.fdtype))
        params = {'raw_theta': Parameter(raw_theta0, requires_grad=self.train_lambdas)}
        self.register_params(params=params)

        # Define prior    
        self.register_prior("raw_theta_prior", NormalPrior(0, self.tau),
                            lambda module: module.raw_theta[2:])
        self.register_prior("raw_theta_prior1", NormalPrior(-1, 1),
                            lambda module: module.raw_theta[1])
        self.register_prior("raw_theta_prior0", NormalPrior(-1, 1),
                            lambda module: module.raw_theta[0])
    
    def define_lambdas_monotonic_decay(self):
        # Set parametrization based on monotonic differences
        raw_theta0 = torch.zeros(self.s).to(self.fdtype)
        if self.log_lambdas0 is None:
            raw_theta0[0] = 0
            raw_theta0[0] = np.log(2) # natural rate of decay
        else:
            raw_theta0[0] = self.log_lambdas0[0]
            delta = self.log_lambdas0[:-1] - self.log_lambdas0[1:] + 1e-6
            if (delta < 0).sum() > 0:
                raise ValueError('log lambdas must decay monotonically')
            logdelta = torch.log(delta)
            raw_theta0[1] = logdelta.mean()
            raw_theta0[2:] = logdelta - raw_theta0[1]
        params = {'raw_theta': Parameter(raw_theta0, requires_grad=self.train_lambdas)}
        self.register_params(params=params)
        
        # Define prior
        self.register_prior("raw_theta_prior", NormalPrior(0, self.tau),
                            lambda module: module.raw_theta[2:])
    
    def define_lambdas(self, tau, lambdas_prior,
                       train_lambdas=True, log_lambdas0=None):
        self.tau = tau
        self.train_lambdas = train_lambdas
        self.log_lambdas0 = None if log_lambdas0 is None else get_tensor(log_lambdas0, dtype=self.fdtype)
         
        if lambdas_prior == 'monotonic_decay': 
            self.define_lambdas_monotonic_decay()
            self.theta_to_log_lambdas = self.theta_to_log_lambdas_monotonic_decay
        elif lambdas_prior == '2nd_order_diff':
            self.define_lambdas_2nd_order()
            self.theta_to_log_lambdas = self.theta_to_log_lambdas_2nd_order_diff
        else:
            msg = 'Prior on lambdas not recognized: {}'.format(lambdas_prior)
            msg += '. Try one of ["monotonic_decay", "2nd_order_diff"]'
            raise ValueError(msg)
     
    def theta_to_log_lambdas_monotonic_decay(self):
        v = self.raw_theta
        z = torch.zeros(1).to(dtype=v.dtype, device=v.device)
        delta_log_lambdas = torch.cat((z, torch.exp(v[1] + v[2:])))
        log_lambdas = self.raw_theta[0] - torch.cumsum(delta_log_lambdas, dim=0)
        return(log_lambdas)
    
    def theta_to_log_lambdas_2nd_order_diff(self):
        log_lambdas = torch.matmul(self.theta_to_log_lda_m, self.raw_theta)
        return(log_lambdas) 

    @property
    def log_lambdas(self):
        log_lambdas = self.theta_to_log_lambdas()
        log_lambdas = torch.cat((get_tensor([-10.], dtype=log_lambdas.dtype,
                                            device=log_lambdas.device), log_lambdas))
        return(log_lambdas)
    
    @property
    def lambdas(self):
        lambdas = torch.exp(self.log_lambdas)
        return(lambdas)
    
    @property
    def log_p(self):
        return(self.raw_log_p)

    def normalize_log_p(self, log_p):
        return(log_p - torch.logsumexp(log_p, 1).unsqueeze(1))
    
    @property
    def p(self):
        return(torch.exp(self.normalize_log_p(self.log_p)))

    @log_lambdas.setter
    def log_lambdas(self, value):
        return self._set_log_lambdas(value)

    @log_p.setter
    def log_p(self, value):
        return self._set_log_p(value)
    
    def _forward(self, x1, x2, lambdas, log_p, diag=False):
        coeffs = self.coeffs.to(dtype=lambdas.dtype)
        c_ki = torch.matmul(coeffs, lambdas)
        coeff_signs = torch.ones_like(c_ki)
        coeff_signs[c_ki < 0] = -1
        log_c_ki = torch.log(torch.abs(c_ki))
        
        log_p = self.normalize_log_p(log_p)
        log_p_flat = torch.flatten(log_p[:, :-1])

        # Init first power
        M = torch.diag(log_p_flat)
        if diag:
            kernel = coeff_signs[0] * torch.exp(log_c_ki[0]-(torch.matmul(x1, M) * x2).sum(1))
        else:
            kernel = coeff_signs[0] * torch.exp(log_c_ki[0]-self.inner_product(x1, x2, M))
            kernel *= torch.matmul(x1, x2.T) == self.l
        
        # Add the remaining powers        
        for power in range(1, self.s):
            log_factors = torch.stack([self.log_odds[power] - log_p_flat,
                                       torch.zeros_like(log_p_flat)], 1)
            log_factors = torch.logsumexp(log_factors, 1)
            M = torch.diag(log_factors)
            m = self.inner_product(x1, x2, M, diag=diag)
            kernel += coeff_signs[power] * torch.exp(log_c_ki[power] + self.log_scaling_factors[power] + m)
        return(kernel)
    
    def forward(self, x1, x2, diag=False, **params):
        kernel = self._forward(x1, x2, lambdas=self.lambdas, log_p=self.log_p,
                               diag=diag)
        return(kernel)
    

class SiteProductKernel(SequenceKernel):
    def __init__(self, n_alleles, seq_length,
                 dtype=torch.float32, **kwargs):
        super().__init__(n_alleles, seq_length, dtype=dtype, **kwargs)
        
        self.define_params()

    def define_params(self):
        params = {'raw_w': Parameter(torch.zeros(self.l), requires_grad=True),
                  'raw_b': Parameter(torch.zeros(1), requires_grad=True),
                  'raw_a': Parameter(torch.log(torch.ones(1)), requires_grad=True)}
        self.register_params(params=params, constraints={})
        
    @property
    def beta(self):
        beta = torch.exp(self.raw_b)
        return(beta)
    
    @property
    def a(self):
        return(self.raw_a)
    
    @property
    def w(self):
        return(self.raw_w)

    def _forward(self, x1, x2, a, beta, w, diag=False):
        # TODO: make sure diag works here
        ebeta = torch.exp(-beta + a)
        log_factors = torch.log(1 + torch.exp(-beta + a + w))
        log_factors = torch.flatten(torch.stack([log_factors] * self.alpha, axis=0).T)
        M = torch.diag(log_factors)
        m = self.inner_product(x1, x2, M, diag=diag)
        
        distance = self.l - self.inner_product(x1, x2, diag=diag)
        kernel = torch.exp(m) * (1 - ebeta) ** distance 
        return(kernel)
    
    def forward(self, x1, x2, diag=False, **params):
        return(self._forward(x1, x2, a=self.a, beta=self.beta, w=self.w, diag=diag))


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
