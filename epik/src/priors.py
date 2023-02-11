import numpy as np
import torch as torch

from torch.nn import Parameter
from gpytorch.priors.torch_priors import NormalPrior


class KernelParamPrior(object):
    def __init__(self, seq_length, n_alleles, train=True):
        self.l = seq_length
        self.alpha = n_alleles
        self.s = seq_length + 1
        self.train = train
    
    def set(self, kernel):
        self.set_params(kernel)
        self.set_priors(kernel)


class LambdasExpDecayPrior(KernelParamPrior):
    def __init__(self, seq_length, tau=0.2, log_lambdas0=None, train=True):
        super().__init__(seq_length=seq_length, n_alleles=None, train=train)
        self.tau = tau
        self.calc_log_lambdas_to_theta_matrix()
        self.calc_theta_to_log_lambdas_matrix()
        self.log_lambdas0 = log_lambdas0
    
    def calc_log_lambdas_to_theta_matrix(self):
        matrix = torch.zeros((self.l, self.l))
        matrix[0, 0] = 1
        matrix[1, 0] = -1
        matrix[1, 1] = 1
        for i in range(2, self.l):
            matrix[i, i-2] = -1
            matrix[i, i-1] = 2
            matrix[i, i] = -1
        self.log_lambdas_to_theta_matrix = Parameter(matrix, requires_grad=False)
    
    def calc_theta_to_log_lambdas_matrix(self):
        self.theta_to_log_lambdas_matrix = torch.inverse(self.log_lambdas_to_theta_matrix)
        
    def theta_to_log_lambdas(self, theta):
        log_lambdas = torch.matmul(self.theta_to_log_lambdas_matrix, theta)
        return(log_lambdas)
    
    def log_lambdas_to_theta(self, log_lambdas):
        theta = torch.matmul(self.log_lambdas_to_theta_matrix, log_lambdas)
        return(theta)

    def get_theta0(self):
        if self.log_lambdas0 is None:
            raw_theta0 = torch.zeros(self.l)
            raw_theta0[0:2] = -1
        else:
            raw_theta0 = self.log_lambdas_to_theta(self.log_lambdas0)
        return(raw_theta0)

    def set_params(self, kernel):
        raw_theta0 = self.get_theta0()
        theta = {'raw_theta': Parameter(raw_theta0, requires_grad=self.train)}
        kernel.register_params(theta)
    
    def set_priors(self, kernel):
        kernel.register_prior("raw_theta_prior", NormalPrior(0, self.tau),
                              lambda module: module.raw_theta[2:])
        kernel.register_prior("raw_theta_prior1", NormalPrior(-1, 1),
                              lambda module: module.raw_theta[1])
        kernel.register_prior("raw_theta_prior0", NormalPrior(-1, 1),
                              lambda module: module.raw_theta[0])


class LambdasMonotonicDecayPrior(KernelParamPrior):
    def __init__(self, seq_length, tau, log_lambdas0=None, train=True):
        super().__init__(seq_length=seq_length, n_alleles=None, train=train)
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
        kernel.register_prior("raw_theta_prior", NormalPrior(0, self.tau),
                              lambda module: module.raw_theta[2:])
        
class AllelesProbPrior(KernelParamPrior):
    def __init__(self, seq_length, n_alleles, eta=None, beta0=None, train=True):
        super().__init__(seq_length=seq_length, n_alleles=n_alleles, train=train)
        self.eta = eta
        self.beta0 = beta0
        
    def normalize_log_p(self, logp):
        return(logp - torch.logsumexp(logp, 1).unsqueeze(1))
    
    def norm_logp_to_beta(self, logp):
        beta = torch.log(1 - torch.exp(logp)) - logp
        return(beta)
    
    def beta_to_logp(self, beta):
        logp = -beta - torch.log(1 + torch.exp(-beta))
        return(logp)

    def get_logp0(self):
        if self.beta0 is None:
            raw_logp0 = torch.zeros((self.l, self.alpha + 1))
        else:
            raw_logp0 = self.beta_to_logp(self.beta0)
        return(raw_logp0)

    def set_params(self, kernel):
        raw_logp0 = self.get_logp0()
        logp = {'raw_logp': Parameter(raw_logp0, requires_grad=self.train)}
        kernel.register_params(logp)
        
    def set_priors(self, kernel):
        if self.eta is not None:
            kernel.register_prior("raw_logp_prior", NormalPrior(0, self.eta),
                                  lambda module: module.raw_logp)
