import numpy as np
import torch as torch

from torch.nn import Parameter
from gpytorch.priors.torch_priors import NormalPrior
from scipy.special._basic import comb
from epik.src.utils import get_tensor


class KernelParamPrior(object):
    def __init__(self, seq_length, n_alleles, train=True, dtype=torch.float32):
        self.l = seq_length
        self.alpha = n_alleles
        self.s = seq_length + 1
        self.train = train
        self.dtype = dtype
        if n_alleles is not None:
            self.n_genotypes = n_alleles ** seq_length 
    
    def set(self, kernel):
        self.set_params(kernel)
        self.set_priors(kernel)


class LambdasDeltaPrior(KernelParamPrior):
    def __init__(self, seq_length, n_alleles, P, train=True,
                 dtype=torch.float32):
        super().__init__(seq_length=seq_length, n_alleles=n_alleles, train=train,
                         dtype=dtype)
        self.P = P
        self.n_p_faces = comb(self.l, 2) * comb(self.alpha, 2) ** 2 * self.alpha ** (self.l - 2)
        self.m_k = [comb(self.l, k) for k in range(self.l + 1)]
        self.kernel_dimension = np.sum(self.m_k[:P])
        
        log_k_P_combs = torch.log(get_tensor([comb(k, P) for k in range(P, self.l + 1)]))
        self.DP_log_lambda = Parameter(self.P * np.log(n_alleles) + log_k_P_combs,
                                       requires_grad=False)
    
    def theta_to_log_lambdas(self, theta, log_tau=None, kernel=None):
        obj = self if kernel is None else kernel
        log_tau = obj.raw_log_tau if log_tau is None else log_tau
        log_a = np.log(self.n_genotypes - self.kernel_dimension) - 2 * log_tau
        log_lambdas = torch.zeros(self.s)
        log_lambdas[:self.P] = theta
        log_lambdas[self.P:] = - log_a + np.log(self.n_p_faces) - obj.DP_log_lambda
        return(log_lambdas.to(dtype=theta.dtype, device=theta.device))
    
    def get_theta0(self):
        return(torch.zeros(self.P))
    
    def get_log_tau0(self):
        return(torch.zeros(1))

    def set_params(self, kernel):
        raw_log_tau0 = self.get_log_tau0()
        raw_theta0 = self.get_theta0()
        theta = {'raw_theta': Parameter(raw_theta0, requires_grad=self.train),
                 'raw_log_tau': Parameter(raw_log_tau0, requires_grad=self.train),
                 'DP_log_lambda': Parameter(self.DP_log_lambda, requires_grad=False)}
        kernel.register_params(theta)
    
    def set_priors(self, kernel):
        return()


class LambdasFlatPrior(KernelParamPrior):
    def __init__(self, seq_length, log_lambdas0=None, train=True,
                 dtype=torch.float32):
        super().__init__(seq_length=seq_length, n_alleles=None, train=train,
                         dtype=dtype)
        self.log_lambdas0 = log_lambdas0
    
    def theta_to_log_lambdas(self, theta, kernel=None):
        return(theta)
    
    def log_lambdas_to_theta(self, log_lambdas):
        return(log_lambdas)

    def get_params0(self):
        if self.log_lambdas0 is None:
            raw_theta0 = -torch.arange(self.l+1).to(dtype=self.dtype)
            # raw_theta0 = torch.zeros(self.l+1)
        else:
            raw_theta0 = self.log_lambdas0
        return(raw_theta0)

    def set_params(self, kernel):
        raw_theta0 = self.get_params0()
        theta = {'raw_theta': Parameter(raw_theta0, requires_grad=self.train)}
        kernel.register_params(theta)
    
    def set_priors(self, kernel):
        return()


class LambdasExpDecayPrior(KernelParamPrior):
    def __init__(self, seq_length, log_lambdas0=None, train=True,
                 dtype=torch.float32):
        super().__init__(seq_length=seq_length, n_alleles=None, train=train,
                         dtype=dtype)
        self.calc_log_lambdas_theta_matrices()
        self.log_lambdas0 = log_lambdas0
    
    def calc_log_lambdas_theta_matrices(self):
        matrix = torch.zeros((self.l, self.l))
        matrix[0, 0] = 1
        matrix[1, 0] = -1
        matrix[1, 1] = 1
        for i in range(2, self.l):
            matrix[i, i-2] = -1
            matrix[i, i-1] = 2
            matrix[i, i] = -1
        self.log_lambdas_to_theta_matrix = Parameter(matrix, requires_grad=False)
        
    def theta_to_log_lambdas(self, theta, tau=None, kernel=None):
        obj = self if kernel is None else kernel
        tau = torch.exp(obj.raw_tau) if tau is None else tau 
        theta_scaled = torch.zeros_like(theta)
        theta_scaled[:3] = theta[:3] 
        theta_scaled[3:] = tau * theta[3:]
        
        log_lambdas = torch.zeros_like(theta)
        log_lambdas[0] = theta_scaled[0] 
        log_lambdas[1:] = torch.linalg.solve(self.log_lambdas_to_theta_matrix, theta_scaled[1:])
        return(log_lambdas)
    
    def log_lambdas_to_theta(self, log_lambdas, kernel=None):
        obj = self if kernel is None else kernel
        theta = torch.matmul(obj.log_lambdas_to_theta_matrix, log_lambdas)
        return(theta)

    def get_params0(self):
        if self.log_lambdas0 is None:
            raw_theta0 = torch.zeros(self.l+1)
            raw_theta0[1] = 1
            # raw_theta0[2] = 1
            raw_tau0 = self.get_tau0()
        else:
            raw_theta0 = self.log_lambdas_to_theta(self.log_lambdas0)
            raw_tau0 = torch.log(torch.std(raw_theta0[3:]))
        return(raw_theta0, raw_tau0)

    def get_tau0(self):
        return(torch.zeros(1))

    def set_params(self, kernel):
        raw_theta0, raw_tau0 = self.get_params0()
        theta = {'raw_theta': Parameter(raw_theta0, requires_grad=self.train),
                 'raw_tau': Parameter(raw_tau0, requires_grad=self.train),
                 'log_lambdas_to_theta_matrix': self.log_lambdas_to_theta_matrix}
        
        kernel.register_params(theta)
    
    def set_priors(self, kernel):
        kernel.register_prior("raw_theta_prior", NormalPrior(0, 1),
                              lambda module: module.raw_theta[3:])


class LambdasMonotonicDecayPrior(KernelParamPrior):
    # TODO: fix to new parametrization
    def __init__(self, seq_length, tau=None, log_lambdas0=None, train=True,
                 dtype=torch.float32):
        super().__init__(seq_length=seq_length, n_alleles=None, train=train,
                         dtype=dtype)
        self.tau = tau
        self.log_lambdas0 = log_lambdas0
    
    def theta_to_log_lambdas(self, theta):
        a = theta[0]
        b = theta[1]
        x_raw = theta[2:]
        
        z = torch.zeros(1).to(dtype=theta.dtype, device=theta.device)
        delta_log_lambdas = torch.cat((z, torch.exp(b + x_raw)))
        log_lambdas = a - torch.cumsum(delta_log_lambdas, dim=0)
        return(log_lambdas)
    
    def log_lambdas_to_theta(self, log_lambdas):
        theta = torch.zeros(self.s)
        theta[0] = log_lambdas[0]
        delta = self.log_lambdas0[:-1] - self.log_lambdas0[1:] + 1e-6
        if (delta < 0).sum() > 0:
            raise ValueError('log lambdas must decay monotonically')
        logdelta = torch.log(delta)
        theta[1] = logdelta.mean()
        theta[2:] = logdelta - theta[1]
        return(theta)

    def get_theta0(self):
        if self.log_lambdas0 is None:
            raw_theta0 = torch.zeros(self.l)
            raw_theta0[1] = np.log(2)
        else:
            raw_theta0 = self.log_lambdas_to_theta(self.log_lambdas0)
        return(raw_theta0)

    def set_params(self, kernel):
        raw_theta0 = self.get_theta0()
        theta = {'raw_theta': Parameter(raw_theta0, requires_grad=self.train)}
        kernel.register_params(theta)
        
    def set_priors(self, kernel):
        if self.tau is not None:
            kernel.register_prior("raw_theta_prior", NormalPrior(0, self.tau),
                                  lambda module: module.raw_theta[2:])
        
class AllelesProbPrior(KernelParamPrior):
    def __init__(self, seq_length, n_alleles, eta=None, beta0=None, train=True,
                 sites_equal=False, alleles_equal=False, dtype=torch.float32,
                 dummy_allele=False):
        super().__init__(seq_length=seq_length, n_alleles=n_alleles, train=train,
                         dtype=dtype)
        self.eta = eta
        self.sites_equal = sites_equal
        self.alleles_equal = alleles_equal
        self.dummy_allele = dummy_allele
        self.calc_shape()
        self.beta0 = beta0
        
    def calc_shape(self):
        nrows = 1 if self.sites_equal else self.l
        ncols = 1 if self.alleles_equal else self.alpha + int(self.dummy_allele)
        self.shape = (nrows, ncols)

    def resize_logp(self, logp):
        if self.sites_equal:
            ones = torch.ones((self.l, 1), device=logp.device)
            logp = torch.matmul(ones, logp)
            
        if self.alleles_equal:
            if self.dummy_allele:
                log1mp = torch.log(1 - torch.exp(logp)) * torch.ones((self.l, 1))
                logp = logp - np.log(self.alpha)
                ones = torch.ones((1, self.alpha))
                logp = torch.cat([torch.matmul(logp, ones), log1mp], 1)
            else:
                ones = torch.ones((1, self.alpha), dtype=self.dtype)
                logp = torch.matmul(logp, ones)
        
        return(logp)
        
    def normalize_logp(self, logp):
        return(logp - torch.logsumexp(logp, 1).unsqueeze(1))
    
    def norm_logp_to_beta(self, logp):
        beta = torch.log(1 - torch.exp(logp)) - logp
        return(beta)
    
    def beta_to_logp(self, beta):
        logp = beta - torch.log(1 + torch.exp(beta))
        return(logp)

    def get_logp0(self):
        if self.beta0 is None:
            raw_logp0 = torch.zeros(self.shape)-1e-6
        else:
            raw_logp0 = self.beta_to_logp(self.beta0)
        return(raw_logp0)

    def set_params(self, kernel):
        raw_logp0 = self.get_logp0().to(dtype=self.dtype)
        logp = {'raw_logp': Parameter(raw_logp0, requires_grad=self.train)}
        kernel.register_params(logp)
        
    def set_priors(self, kernel):
        if self.eta is not None:
            kernel.register_prior("raw_logp_prior", NormalPrior(0, self.eta),
                                  lambda module: module.raw_logp)


class RhosPrior(KernelParamPrior):
    def __init__(self, seq_length, n_alleles, sites_equal=False,
                 rho0=None, v0=1.1, train=True, dtype=torch.float32,
                 train_constant_component=False):
        super().__init__(seq_length=seq_length, n_alleles=n_alleles, train=train,
                         dtype=dtype)
        self.sites_equal = sites_equal
        self.calc_shape()
        self.train_constant_component = train_constant_component
        
        if rho0 is None:
            self.logit_rho0 = torch.tensor(self.v_to_logit_rho(v0), dtype=self.dtype)
        else:
            self.logit_rho0 = torch.log(rho0 / (1 - rho0)).to(dtype=self.dtype)
    
    def v_to_logit_rho(self, v):
        rho = (np.exp(np.log(v) / self.l) - 1) / (self.alpha - 1)
        logit = [np.log(rho / (1 - rho))]
        if not self.sites_equal:
            logit = logit * self.l
        return(logit)
        
    def calc_shape(self):
        self.shape = (1 if self.sites_equal else self.l,)
        
    def get_logit_rho0(self):
        if self.sites_equal:
            return(self.logit_rho0 * torch.ones(self.shape, dtype=self.dtype))
        else:
            return(self.logit_rho0)
    
    def get_raw_rho_c0(self):
        return(torch.zeros((1,), dtype=self.dtype))
    
    def set_params(self, kernel):
        if self.sites_equal:
            params = {'raw_rho': Parameter(self.get_logit_rho0(), requires_grad=self.train),
                      'raw_rho_c': Parameter(self.get_raw_rho_c0(), requires_grad=self.train_constant_component)}
        else:
            logit_rho = self.get_logit_rho0()
            logit_rho_mu, logit_rho_sd = logit_rho.mean(), logit_rho.std() + 0.01
            params = {'raw_rho_c': Parameter(self.get_raw_rho_c0(), requires_grad=self.train_constant_component),
                      'raw_mu': Parameter(logit_rho_mu, requires_grad=self.train),
                      'raw_sigma': Parameter(torch.log(logit_rho_sd), requires_grad=self.train),
                      'raw_rho': Parameter((logit_rho - logit_rho_mu) / (logit_rho_sd),
                                           requires_grad=self.train),}
        kernel.register_params(params=params, constraints={})
    
    def calc_rho(self, raw_rho, mu=None, sigma=None):
        if self.sites_equal:
            ones = torch.ones((self.l, 1))
            logit_rho = torch.matmul(ones, raw_rho)
        elif mu is not None and sigma is not None:
            logit_rho = mu + sigma * raw_rho
        else:
            msg = 'mu and sigma must be provided for `sites_equal=False`'
            raise ValueError(msg)
        rho = torch.exp(logit_rho) / (1 + torch.exp(logit_rho))
        return(rho)
    
    def get_rho(self, kernel):
        if self.sites_equal:
            rho = self.calc_rho(kernel.raw_rho)
        else:
            rho = self.calc_rho(kernel.raw_rho, mu=kernel.raw_mu,
                                sigma=torch.exp(kernel.raw_sigma))
        return(rho)

    def get_rho_c(self, kernel):
        return(torch.exp(kernel.raw_rho_c))

    def set_priors(self, kernel):
        if not self.sites_equal:
            kernel.register_prior("raw_rho_prior", NormalPrior(0, 1),
                                  lambda module: module.raw_rho)
