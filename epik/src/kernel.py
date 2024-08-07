import numpy as np
import torch as torch

from itertools import combinations
from scipy.special._basic import comb
from torch.nn import Parameter
from torch.distributions.transforms import CorrCholeskyTransform

from pykeops.torch import LazyTensor
from gpytorch.settings import max_cholesky_size
from gpytorch.kernels.kernel import Kernel
from gpytorch.lazy.lazy_tensor import delazify
from linear_operator.operators import (KernelLinearOperator, DiagLinearOperator,
                                       MatmulLinearOperator)

from epik.src.utils import get_tensor, log1mexp, log_comb, calc_decay_rates


def get_constant_linop(c, shape, device, dtype):
    v1 = torch.ones(size=(shape[0], 1)).to(device=device, dtype=dtype)
    v2 = c * torch.ones(size=(1, shape[1])).to(device=device, dtype=dtype)
    return(MatmulLinearOperator(v1, v2))
    

class SequenceKernel(Kernel):
    def __init__(self, n_alleles, seq_length, binary=False,
                 dtype=torch.float32, use_keops=False, **kwargs):
        self.alpha = n_alleles

        if binary and n_alleles != 2:
            raise ValueError('binary encoding can only be used with 2 alleles')
        
        self.binary = binary
        self.l = seq_length
        self.s = self.l + 1
        self.t = self.l * self.alpha
        self.fdtype = dtype
        self.use_keops = use_keops
        self.logn = self.l * np.log(self.alpha)
        self.logam1 = np.log(self.alpha - 1)
        super().__init__(**kwargs)
        
    def zeros_like(self, x):
        return(torch.zeros(x.shape).to(dtype=x.dtype, device=x.device))
    
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
    
    def s_to_d(self, s):
        if self.binary:
            d = self.l / 2. - 0.5 * s
        else:
            d = float(self.l) - s
        return(d)

    def calc_hamming_distance(self, x1, x2, diag=False):
        s = self.inner_product(x1, x2, diag=diag)
        d = self.s_to_d(s)
        return(d)
    
    def calc_hamming_distance_linop(self, x1, x2, scale=1., shift=0.):
        shape = (x1.shape[0], x2.shape[0])
        if self.binary:
            d = MatmulLinearOperator(x1, -scale * x2.T / 2)
            d += get_constant_linop(shift + scale * self.l / 2.0, shape, x1.device, x1.dtype)
        else:
            d = MatmulLinearOperator(x1, -scale * x2.T)
            d += get_constant_linop(shift + scale * self.l, shape, x1.device, x1.dtype)
        return(d)
    
    def calc_hamming_distance_keops(self, x1, x2):
        if self.binary:
            x1_ = LazyTensor(x1[..., :, None, :] / np.sqrt(2))
            x2_ = LazyTensor(x2[..., None, :, :] / np.sqrt(2))
            s = (x1_ * x2_).sum(-1)
            d = self.l / 2. - s
        else:
            x1_ = LazyTensor(x1[..., :, None, :])
            x2_ = LazyTensor(x2[..., None, :, :])
            s = (x1_ * x2_).sum(-1)
            d = float(self.l) - s
        return(d)
    
    def _constant_linop(self, x1, x2):
        z1 = torch.ones((x1.shape[0], 1), device=x1.device, dtype=x1.dtype)
        z2 = torch.ones((1, x2.shape[0]), device=x2.device, dtype=x2.dtype)
        c = MatmulLinearOperator(z1, z2)
        return(c)
    
    def forward(self, x1, x2, diag=False, **kwargs):
        if diag:
            kernel = self._nonkeops_forward(x1, x2, diag=True, **kwargs)
        else:
            max_size = max_cholesky_size.value()
            if self.use_keops or (x1.size(-2) > max_size or x2.size(-2) > max_size):
                kernel = self._keops_forward(x1, x2, **kwargs)
            else:
                try:
                    kernel = self._nonkeops_forward(x1, x2, diag=False, **kwargs)
                except RuntimeError:
                    torch.cuda.empty_cache()
                    kernel = self._keops_forward(x1, x2, **kwargs)
        return(kernel)
    

class AdditiveHeteroskedasticKernel(SequenceKernel):
    @property
    def is_stationary(self) -> bool:
        return self.base_kernel.is_stationary

    def __init__( self, base_kernel, n_alleles=None, seq_length=None,
                  log_ds0=None, a=0.5, **kwargs):
        if base_kernel.active_dims is not None:
            kwargs["active_dims"] = base_kernel.active_dims
        
        if hasattr(base_kernel, 'alpha'):
            n_alleles = base_kernel.alpha
        else:
            if n_alleles is None:
                msg = 'If the base kernel does not have n_alleles attribute, '
                msg += 'it should be provided'
                raise ValueError(msg)
        
        if hasattr(base_kernel, 'l'):
            seq_length = base_kernel.l
        else:
            if seq_length is None:
                msg = 'If the base kernel does not have seq_length attribute, '
                msg += 'it should be provided'
                raise ValueError(msg)
        
        super().__init__(n_alleles, seq_length, **kwargs)
        self.log_ds0 = log_ds0
        self.a = a
        self.set_params()
        self.base_kernel = base_kernel

    def set_params(self):
        theta = torch.zeros((self.l, self.alpha)) if self.log_ds0 is None else self.log_ds0
        params = {'theta': Parameter(theta, requires_grad=True),
                  'theta0': Parameter(5 * torch.ones((1,)), requires_grad=True)}
        self.register_params(params)
        
    def get_theta(self):
        t = self.theta
        return(t - t.mean(1).unsqueeze(1))
    
    def get_theta0(self):
        return(self.theta0)
    
    def f(self, x, theta0, theta, a=0, b=1):
        phi = theta0 + (x * theta.reshape(1, 1, self.l * self.alpha)).sum(-1)
        r = a + (b - a) * torch.exp(phi) / (1 + torch.exp(phi))
        return(r)
    
    def forward(self, x1, x2, last_dim_is_batch=False, diag=False, **params):
        orig_output = self.base_kernel.forward(x1, x2, diag=diag,
                                               last_dim_is_batch=last_dim_is_batch,
                                               **params)
        theta0, theta = self.get_theta0(), self.get_theta()
        f1 = self.f(x1, theta0, theta, a=self.a).T
        f2 = self.f(x2, theta0, theta, a=self.a)
        
        if last_dim_is_batch:
            f1 = f1.unsqueeze(-1)
            f2 = f2.unsqueeze(-1)
        if diag:
            f1 = f1.unsqueeze(-1)
            f2 = f2.unsqueeze(-1)
            return(f1 * f2 * delazify(orig_output))
        else:
            return(f1 * f2 * orig_output)

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
        return self.base_kernel.prediction_strategy(train_inputs, train_prior_dist, train_labels, likelihood)


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
            exp_coeffs = torch.zeros((n_components, 1))
            exp_coeffs[0] = np.exp(self.log_var0)
            exp_coeffs[1] = -np.exp(self.log_var0) / self.l
            log_lambdas0 = np.log(torch.linalg.solve_triangular(c_p, exp_coeffs, upper=True)+1e-10)
            # log_lambdas0 = torch.zeros(n_components)
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
                         correlation=True,
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
    
    def get_decay_rates(self, positions=None):
        decay_rates = calc_decay_rates(self.logit_rho.detach().numpy(),
                                       self.log_p.detach().numpy(),
                                       sqrt=False, positions=positions).mean(1)
        return(decay_rates)
    

class JengaKernel(RhoPiKernel):
    is_stationary = True
    def __init__(self, n_alleles, seq_length, **kwargs):
        super().__init__(n_alleles, seq_length,
                         correlation=True, train_p=True, train_var=True,
                         **kwargs)
    
    def get_decay_rates(self, alleles=None, positions=None):
        decay_rates = calc_decay_rates(self.logit_rho.detach().numpy(),
                                       self.log_p.detach().numpy(),
                                       sqrt=True, alleles=alleles, positions=positions)
        return(decay_rates)


class ExponentialKernel(ConnectednessKernel):
    def __init__(self, n_alleles, seq_length, **kwargs):
        super().__init__(n_alleles, seq_length, common_rho=True, **kwargs)
    
    def get_decay_rate(self):
        decay_rate = calc_decay_rates(self.logit_rho.detach().numpy(),
                                      self.log_p.detach().numpy(),
                                      sqrt=False).values.mean()
        return(decay_rate)
        

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
