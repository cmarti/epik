import numpy as np
import torch as torch

from itertools import combinations
from scipy.special._basic import comb
from torch.nn import Parameter
from pykeops.torch import LazyTensor
from torch.distributions.transforms import CorrCholeskyTransform
from linear_operator.operators import KernelLinearOperator, DiagLinearOperator

from epik.src.utils import get_tensor, log1mexp, log_comb
from epik.src.kernel.base import SequenceKernel


class LowOrderKernel(SequenceKernel):
    is_stationary = True
    def __init__(self, n_alleles, seq_length, log_lambdas0=None,
                 log_var0=2., **kwargs):
        super().__init__(n_alleles, seq_length, **kwargs)
        self.log_lambdas0 = log_lambdas0
        self.log_var0 = log_var0
        self.set_params()
    
    def calc_log_lambdas0(self, c_p):
        n_components = c_p.shape[0]
        if self.log_lambdas0 is None:
            # # Match linearly decaying correlations with specified variance
            # exp_coeffs = torch.zeros((n_components, 1))
            # exp_coeffs[0] = np.exp(self.log_var0)
            # exp_coeffs[1] = -np.exp(self.log_var0) / self.l
            # log_lambdas0 = np.log(torch.linalg.solve_triangular(c_p, exp_coeffs, upper=True)+1e-10)
            log_lambdas0 = np.zeros(n_components)
        else:
            log_lambdas0 = self.log_lambdas0

        log_lambdas0 = log_lambdas0.reshape((n_components, 1))
        return(log_lambdas0)
    
    def calc_c_d(self, log_lambdas):
        c_d = self.c_p @ torch.exp(log_lambdas)
        return(c_d)
    
    def get_coeffs(self):
        return(self.calc_c_d(self.log_lambdas))

    def set_params(self):
        c_p = self.calc_c_p()
        log_lambdas0 = self.calc_log_lambdas0(c_p)
        params = {'log_lambdas': Parameter(log_lambdas0.to(dtype=self.dtype), requires_grad=True),
                  'c_p': Parameter(c_p.to(dtype=self.dtype), requires_grad=False)}
        self.register_params(params)
    
    
class AdditiveKernel(LowOrderKernel):
    def calc_c_p(self):
        a, l = self.alpha, self.l
        c_p = torch.tensor([[1., l * (a - 1)],
                            [0,          -a]])
        return(c_p)  
    
    def forward(self, x1, x2, diag=False, **kwargs):
        coeffs = self.get_coeffs()
        kernel = self.calc_hamming_distance_linop(x1, x2,
                                                  scale=coeffs[1],
                                                  shift=coeffs[0])
        return(kernel)


class PairwiseKernel(LowOrderKernel):
    def calc_c_p(self):
        a, l = self.alpha, self.l
        c13 = a * l - 0.5 * a ** 2 * l - 0.5 * l - a * l ** 2 + 0.5 * a ** 2 * l ** 2 + 0.5 * l ** 2
        c23 = -a + 0.5 * a ** 2 + a * l - a ** 2 * l
        c_p = torch.tensor([[1, l * (a - 1),          c13],
                            [0,          -a,          c23],
                            [0,           0, 0.5 * a ** 2]])
        return(c_p)

    def d_to_cov(self, d, coeffs):
        kernel = coeffs[0] + coeffs[1] * d + coeffs[2] * d * d
        return(kernel)
    
    def _nonkeops_forward(self, x1, x2, diag=False, **kwargs):
        coeffs = self.get_coeffs()
        d = self.calc_hamming_distance(x1, x2)
        return(self.d_to_cov(d, coeffs))

    def _covar_func(self, x1, x2, coeffs, **kwargs):
        d = self.calc_hamming_distance_keops(x1, x2)
        return(self.d_to_cov(d, coeffs))
    
    def _keops_forward(self, x1, x2, **kwargs):
        coeffs = self.get_coeffs()
        kernel = KernelLinearOperator(x1, x2, covar_func=self._covar_func,
                                      coeffs=coeffs, **kwargs)
        return(kernel)    
    

class VarianceComponentKernel(SequenceKernel):
    is_stationary = True
    def __init__(self, n_alleles, seq_length, log_lambdas0=None,
                 log_var0=2., max_k=None, **kwargs):
        super().__init__(n_alleles, seq_length, **kwargs)
        self.use_polynomial = max_k is not None
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
        log_w_kd += log_comb(self.l-d, k-q) #- self.logn
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
        hamming_dist = self.calc_hamming_distance(x1, x2, diag=diag).to(dtype=torch.long)
        kernel = w_d[0] * (hamming_dist == 0)
        for d in range(1, self.s):
            kernel += w_d[d] * (hamming_dist == d)
        return(kernel)
    
    def get_c_d(self):
        c_d = self.calc_c_d(self.c_p_sign, self.log_c_p, self.log_lambdas)
        return(c_d)

    def _nonkeops_forward_polynomial_d(self, x1, x2, diag=False, **kwargs):
        c_d = self.get_c_d()
        
        d = self.calc_hamming_distance(x1, x2, diag=diag)
        k = torch.full_like(d, fill_value=c_d[0].item())
        for i in range(1, self.max_k + 1):
            k += c_d[i] * d ** i
        return(k)

    def _nonkeops_forward(self, x1, x2, diag=False, **kwargs):
        if self.use_polynomial:
            return(self._nonkeops_forward_polynomial_d(x1, x2, diag=diag, **kwargs))
        else:
            return(self._nonkeops_forward_hamming_class(x1, x2, diag=diag, **kwargs))
    
    def _covar_func(self, x1, x2, c_d, **kwargs):
        d = self.calc_hamming_distance_keops(x1, x2)
        
        k = c_d[0]
        for i in range(1, self.max_k + 1):
            k += c_d[i] * d ** i
        return(k)
        
    def _keops_forward(self, x1, x2, **kwargs):
        c_d = self.get_c_d()
        return(KernelLinearOperator(x1, x2, covar_func=self._covar_func,
                                    c_d=c_d, **kwargs))
        
class ThreeWayKernel(VarianceComponentKernel):
    def __init__(self, n_alleles, seq_length, **kwargs):
        super().__init__(n_alleles, seq_length, max_k=3, **kwargs)
    

class RhoPiKernel(SequenceKernel):
    def __init__(self, n_alleles, seq_length,
                 logit_rho0=None, log_p0=None, log_var0=None, 
                 train_p=True, train_var=False,
                 common_rho=False, correlation=False,
                 random_init=False,
                 **kwargs):
        super().__init__(n_alleles, seq_length, **kwargs)
        self.logit_rho0 = logit_rho0
        self.random_init = random_init
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
        if self.logit_rho0 is None:
            shape = (1, 1) if self.common_rho else (self.l, 1)
            t = np.exp(-2 / self.l * np.log(10.))
            v = np.log((1 - t) / (self.alpha * t))
            logit_rho0 = torch.full(shape, v, dtype=self.dtype) if self.logit_rho0 is None else self.logit_rho0 
            if self.random_init:
                logit_rho0 = torch.normal(logit_rho0, std=1.)
        else:
            logit_rho0 = self.logit_rho0
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

        if diag:
            min_size = min(x1.shape[0], x2.shape[0])
            log_kernel = constant + (x1[:min_size, :] * x2[:min_size, :] * factors).sum(1)
        else:
            log_kernel = constant + x1 @ (x2 * factors).T
        
        if self.correlation:
            log_sd1 = 0.5 * (x1 @ log_one_p_eta_rho)
            log_sd2 = 0.5 * (x2 @ log_one_p_eta_rho)
            if not diag:
                log_sd2 = log_sd2.reshape((1, x2.shape[0]))
            log_kernel = log_kernel - log_sd1 - log_sd2
        
        kernel = torch.exp(log_kernel)
        return(kernel)
    
    def _covar_func(self, x1, x2, constant, **kwargs):
        x1_ = LazyTensor(x1[..., :, None, :])
        x2_ = LazyTensor(x2[..., None, :, :])
        kernel = ((x1_ * x2_).sum(-1) + constant).exp()
        return(kernel)
    
    def _keops_forward(self, x1, x2, **kwargs):
        # TODO: introduce constants before exponentiation in covar_func
        constant, factors, log_one_p_eta_rho = self.get_factors()
        f = factors.reshape(1, self.t)
        kernel = KernelLinearOperator(x1, x2 * f, 
                                      covar_func=self._covar_func,
                                      constant=constant, **kwargs)
        
        if self.correlation:
            log_one_p_eta_rho = log_one_p_eta_rho.reshape(1, self.t)
            sd1_inv_D = DiagLinearOperator(torch.exp(-0.5 * (x1 * log_one_p_eta_rho).sum(1)))
            sd2_inv_D = DiagLinearOperator(torch.exp(-0.5 * (x2 * log_one_p_eta_rho).sum(1)))
            kernel = sd1_inv_D @ kernel @ sd2_inv_D

        return(kernel)
    
    
class ConnectednessKernel(RhoPiKernel):
    is_stationary = True
    def __init__(self, n_alleles, seq_length, **kwargs):
        super().__init__(n_alleles, seq_length,
                         train_p=False, train_var=True,
                         **kwargs)
    
    def _nonkeops_forward_binary(self, x1, x2, diag=False, **params):
        rho = torch.exp(self.logit_rho) / (1 + torch.exp(self.logit_rho))
        kernel = torch.prod(1 + x1.unsqueeze(0) * (x2 * rho.reshape((1, self.l))).unsqueeze(1), axis=2)
        if self.correlation:
            kernel  = kernel / torch.prod(1 + rho)
        return(kernel)
    
    def _covar_func_binary2(self, x1, x2, **kwargs):
        x1_ = LazyTensor(x1[..., :, None, :])
        x2_ = LazyTensor(x2[..., None, :, :])
        kernel = (1 + x1_ * x2_).log().sum(-1).exp()
        return(kernel)
    
    def _keops_forward_binary2(self, x1, x2, **kwargs):
        rho = torch.exp(self.logit_rho) / (1 + torch.exp(self.logit_rho))
        kernel = KernelLinearOperator(x1, x2 * rho.reshape((1, self.l)),
                                      covar_func=self._covar_func_binary, **kwargs)
        if self.correlation:
            c = 1 / torch.prod(1 + rho)
            kernel = c * kernel
        return(kernel)
    
    def _covar_func_binary(self, x1, x2, constant, f, **kwargs):
        x1_ = LazyTensor(x1[..., :, None, :])
        x2_ = LazyTensor(x2[..., None, :, :])
        f = f.reshape((1, 1, self.l))
        kernel = (constant + 0.5 * ((x1_ * x2_ * f).sum(-1) + f.sum())).exp()
        return(kernel)
    
    def _keops_forward_binary(self, x1, x2, **kwargs):
        log1mrho = self.get_log_one_minus_rho()
        log_rho = self.logit_rho + log1mrho
        log_one_p_rho = torch.logaddexp(self.zeros_like(log_rho), log_rho)
        factors = log_one_p_rho - log1mrho
        
        constant = log1mrho.sum()
        if self.correlation:
            constant -= log_one_p_rho.sum()
        if self.common_rho:
            constant *= self.l
        constant += self.log_var
        
        f = factors.reshape(1, self.l)
        kernel = KernelLinearOperator(x1, x2, covar_func=self._covar_func_binary,
                                      constant=constant, f=f, **kwargs)
        return(kernel)
    
    def _keops_forward(self, x1, x2, **kwargs):
        if self.binary:
            return(self._keops_forward_binary(x1, x2, **kwargs))
        else:
            return(super()._keops_forward(x1, x2, **kwargs))
    
    def _nonkeops_forward(self, x1, x2, **kwargs):
        if self.binary:
            return(self._nonkeops_forward_binary(x1, x2, **kwargs))
        else:
            return(super()._nonkeops_forward(x1, x2, **kwargs))
    

class JengaKernel(RhoPiKernel):
    is_stationary = True
    def __init__(self, n_alleles, seq_length, **kwargs):
        super().__init__(n_alleles, seq_length,
                         correlation=True, train_p=True, train_var=True,
                         **kwargs)


class ExponentialKernel(ConnectednessKernel):
    def __init__(self, n_alleles, seq_length, **kwargs):
        super().__init__(n_alleles, seq_length, common_rho=True, **kwargs)
        

class GeneralProductKernel(SequenceKernel):
    is_stationary = True
    def __init__(self, n_alleles, seq_length, theta0=None, **kwargs):
        super().__init__(n_alleles, seq_length, **kwargs)
        self.dim = int(comb(n_alleles, 2))
        self.theta0 = theta0
        self.set_params()
        self.theta_to_L = CorrCholeskyTransform()
    
    def calc_theta0(self):
        if self.theta0 is not None:
            theta0 = self.theta0
        else:
            theta0 = torch.zeros((self.l, self.dim), dtype=self.dtype)
        return(theta0)
    
    def theta_to_covs(self, theta):
        Ls = [self.theta_to_L(theta[i]) for i in range(self.l)]
        covs = torch.stack([L @ L.T for L in Ls], axis=2)
        return(covs)
    
    def set_params(self):
        theta0 = self.calc_theta0()
        params = {'theta': Parameter(theta0, requires_grad=True)}
        self.register_params(params)

    def get_covs(self):
        return(self.theta_to_covs(self.theta))
    
    def forward(self, x1, x2, diag=False, **kwargs):
        covs = self.get_covs()
        K = x1[:, :self.alpha] @ covs[:, :, 0] @ x2[:, :self.alpha].T
        for i in range(1, self.l):
            start, end = i * self.alpha, (i+1) * self.alpha
            K *= x1[:, start:end] @ covs[:, :, i] @ x2[:, start:end].T
        return(K)


#################
# Extra kernels #
#################

class AddRhoPiKernel(RhoPiKernel):
    def __init__(self, n_alleles, seq_length,
                 logit_rho0=None, log_p0=None, log_var0=None, 
                 train_p=False, train_var=True,
                 common_rho=False, correlation=False,
                 random_init=False,
                 **kwargs):
        super().__init__(n_alleles, seq_length, **kwargs)
        self.logit_rho0 = logit_rho0
        self.random_init = random_init
        self.log_p0 = log_p0
        self.log_var0 = log_var0
        self.train_p = train_p
        self.train_var = train_var
        self.correlation = correlation
        self.common_rho = common_rho
        self.set_params()
        
        logl = torch.log(torch.tensor([seq_length], dtype=self.dtype, device=self.device))
        self.register_params({'logl': Parameter(logl, requires_grad=False)})
    
    def _nonkeops_forward(self, x1, x2, diag=False, **params):
        constant, factors, log_one_p_eta_rho = self.get_factors()
        factors = factors.reshape(1, self.t)
        log_one_p_eta_rho = log_one_p_eta_rho.reshape(self.t, 1)
        
        if diag:
            min_size = min(x1.shape[0], x2.shape[0])
            x1_x2 = x1[:min_size, :] * x2[:min_size, :]
            log_s = torch.log((x1_x2).sum(1)) - self.logl
            log_kernel = constant + (x1_x2 * factors).sum(1) + log_s
        else:
            log_s = torch.log(x1 @ x2.T) - self.logl
            log_kernel = constant + x1 @ (x2 * factors).T + log_s
        
        if self.correlation:
            log_sd1 = 0.5 * (x1 @ log_one_p_eta_rho)
            log_sd2 = 0.5 * (x2 @ log_one_p_eta_rho)
            if diag:
                log_sd2 = log_sd2.reshape((1, x2.shape[0]))
            log_kernel = log_kernel - log_sd1 - log_sd2
        
        kernel = torch.exp(log_kernel)
        return(kernel)
    
    def _covar_func(self, x1, x2, f, **kwargs):
        x1_ = LazyTensor(x1[..., :, None, :])
        x2_ = LazyTensor(x2[..., None, :, :])
        kernel = ((x1_ * f[:, None] * x2_).sum(-1) + (x1_ * x2_).sum(-1).log()).exp()
        return(kernel)
    
    def _keops_forward(self, x1, x2, **kwargs):
        # TODO: introduce constants before exponentiation in covar_func
        constant, factors, log_one_p_eta_rho = self.get_factors()
        f = factors.reshape(1, self.t)
        log_one_p_eta_rho = log_one_p_eta_rho.reshape(1, self.t)
        c = torch.exp(constant - self.logl)
        
        kernel = c * KernelLinearOperator(x1, x2, covar_func=self._covar_func, f=f, **kwargs)
        if self.correlation:
            sd1_inv_D = DiagLinearOperator(torch.exp(-0.5 * (x1 * log_one_p_eta_rho).sum(1)))
            sd2_inv_D = DiagLinearOperator(torch.exp(-0.5 * (x2 * log_one_p_eta_rho).sum(1)))
            kernel = sd1_inv_D @ kernel @ sd2_inv_D
        return(kernel)


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
