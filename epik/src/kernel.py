import numpy as np
import torch as torch
from gpytorch.kernels import Kernel
from linear_operator.operators import KernelLinearOperator
from pykeops.torch import LazyTensor
from scipy.special import comb
from torch.distributions.transforms import CorrCholeskyTransform
from torch.nn import Parameter

from epik.src.utils import (
    KrawtchoukPolynomials,
    log1mexp,
    inner_product,
    HammingDistanceCalculator,
)


class SequenceKernel(Kernel):
    def __init__(self, n_alleles, seq_length, use_keops=False, **kwargs):
        self.n_alleles = n_alleles
        self.seq_length = seq_length
        self.lp1 = seq_length + 1
        self.n_features = seq_length * n_alleles

        self.logn = self.seq_length * np.log(self.n_alleles)
        self.logam1 = np.log(self.n_alleles - 1)
        self.use_keops = use_keops
        super().__init__(**kwargs)

    def select_site(self, x, site):
        idx = torch.arange(site * self.n_alleles, (site + 1) * self.n_alleles)
        return x.index_select(-1, idx.to(device=x.device))

    def register_params(self, params={}, constraints={}):
        self.params = params
        for param_name, param in params.items():
            self.register_parameter(name=param_name, parameter=param)

        for param_name, constraint in constraints.items():
            self.register_constraint(param_name, constraint)

    def calc_hamming_distance(self, x1, x2, diag=False, keops=False):
        if diag or not keops:
            s = inner_product(x1, x2, diag=diag)
            d = float(self.seq_length) - s
        else:
            x1_ = LazyTensor(x1[..., :, None, :])
            x2_ = LazyTensor(x2[..., None, :, :])
            s = (x1_ * x2_).sum(-1)
            d = float(self.seq_length) - s
        return d

    def forward(self, x1, x2, diag=False, **kwargs):
        if diag:
            kernel = self._nonkeops_forward(x1, x2, diag=True, **kwargs)

        else:
            if self.use_keops:
                kernel = self._keops_forward(x1, x2, **kwargs)

            else:
                try:
                    kernel = self._nonkeops_forward(x1, x2, diag=False, **kwargs)
                except RuntimeError:  # Memory error
                    self.use_keops = True
                    torch.cuda.empty_cache()
                    kernel = self._keops_forward(x1, x2, **kwargs)

        return kernel


class BaseVarianceComponentKernel(SequenceKernel):
    is_stationary = True

    def __init__(
        self, n_alleles, seq_length, log_lambdas0=None, cov0=None, ns0=None, **kwargs
    ):
        super().__init__(n_alleles, seq_length, **kwargs)
        self.log_lambdas0 = log_lambdas0
        self.cov0 = cov0
        self.ns0 = ns0
        self.ws = KrawtchoukPolynomials(n_alleles, seq_length, max_k=self.max_k)
        self.set_params()

    def get_log_lambdas0(self):
        if self.log_lambdas0 is None:
            if self.cov0 is None:
                log_lambdas0 = torch.zeros(self.max_k + 1)
            else:
                lambdas0 = self.ws.calc_lambdas(self.cov0, self.ns0)
                if torch.any(lambdas0 < 0.0):
                    log_lambdas0 = torch.zeros(self.max_k + 1)
                else:
                    log_lambdas0 = torch.log(lambdas0 + 1e-6)
        else:
            log_lambdas0 = self.log_lambdas0

        log_lambdas0 = log_lambdas0.to(dtype=self.dtype)
        return log_lambdas0

    def set_params(self):
        c_bk = self.calc_c_bk()
        log_lambdas0 = self.get_log_lambdas0()
        theta = torch.Tensor([[-np.log(0.1) / self.seq_length]])
        params = {
            "log_lambdas": Parameter(log_lambdas0, requires_grad=True),
            "theta": Parameter(theta, requires_grad=False),
            "c_bk": Parameter(c_bk, requires_grad=False),
        }
        self.register_params(params)

    def calc_c_b(self, log_lambdas):
        c_b = self.c_bk @ torch.exp(log_lambdas)
        return c_b

    def get_c_b(self):
        return self.calc_c_b(self.log_lambdas)

    def get_basis(self, d, keops=False):
        if keops:
            yield (1.0)
        else:
            yield (torch.ones_like(d))

        d_power = d
        for _ in range(1, self.max_k):
            yield (d_power)
            d_power = d_power * d
        yield (d_power)

    def d_to_cov(self, d, c_b, keops=False):
        basis = self.get_basis(d, keops=keops)
        b0 = next(basis)
        kernel = c_b[0] * b0
        for b_i, c_i in zip(basis, c_b[1:]):
            kernel += c_i * b_i
        return kernel

    def _nonkeops_forward(self, x1, x2, diag=False, **kwargs):
        c_b = self.get_c_b()
        d = self.calc_hamming_distance(x1, x2, diag=diag)
        return self.d_to_cov(d, c_b, keops=False)

    def _covar_func(self, x1, x2, c_b, **kwargs):
        d = self.calc_hamming_distance(x1, x2, keops=True)
        return self.d_to_cov(d, c_b, keops=True)

    def _keops_forward(self, x1, x2, **kwargs):
        c_b = self.get_c_b()
        kernel = KernelLinearOperator(
            x1, x2, covar_func=self._covar_func, c_b=c_b, **kwargs
        )
        return kernel


class AdditiveKernel(BaseVarianceComponentKernel):
    """
    Kernel function for additive functions on sequence space, where the
    covariance between two sequences is linear in the Hamming distance
    that separates them, with parameters determined by the variance
    explained by the constant and additive components.

    .. math::
        K(x, y) = c_0 + c_1 * d(x, y)
        c_0 = \lambda_0 + \ell * (\alpha - 1) * \lambda_1
        c_1 = -\alpha * \lambda_1

    When instantiated at a set of one-hot sequence embeddings `x1` and `x2`,
    it returns a linear operator that performs fast matrix-vector products
    without explicitly building the full covariance matrix.

    """

    @property
    def max_k(self):
        return 1

    def calc_c_bk(self):
        a, sl = self.n_alleles, self.seq_length
        c_bk = torch.tensor([[1.0, sl * (a - 1)], [0, -a]])
        return c_bk

    def forward(self, x1, x2, diag=False, **kwargs):
        c_b = self.get_c_b()
        if diag:
            d = self.calc_hamming_distance(x1, x2, diag=True)
            kernel = c_b[0] + c_b[1] * d
        else:
            calc_d = HammingDistanceCalculator(
                self.seq_length, scale=c_b[1], shift=c_b[0]
            )
            kernel = calc_d(x1, x2)
        return kernel


class PairwiseKernel(BaseVarianceComponentKernel):
    """
    Kernel function for additive functions on sequence space, where the
    covariance between two sequences is quadratic in the Hamming distance
    that separates them, with coefficients determined by the variance
    explained by the constant, additive and pairwise components.

    .. math::
        K(x, y) = c_0 + c_1 * d(x, y) + c_2 * d(x, y)^2

    These coefficients result from expanding the Krawtchouk polynomials
    of order 2, as in the additive kernel, and allows computing the covariance
    matrix easily for any number of sequences of any length.
    """

    @property
    def max_k(self):
        return 2

    def get_basis(self, d, keops=False):
        b0 = 1.0 if keops else torch.ones_like(d)
        yield (b0)
        yield (d)
        yield (d.square())

    def calc_c_bk(self):
        a, sl = self.n_alleles, self.seq_length
        c13 = (
            a * sl
            - 0.5 * a**2 * sl
            - 0.5 * sl
            - a * sl**2
            + 0.5 * a**2 * sl**2
            + 0.5 * sl**2
        )
        c23 = -a + 0.5 * a**2 + a * sl - a**2 * sl
        c_bk = torch.tensor([[1, sl * (a - 1), c13], [0, -a, c23], [0, 0, 0.5 * a**2]])
        return c_bk
        # c_sb = torch.tensor([[1, sl,         sl ** 2 ],
        #                     [0,  -1,        -2  * sl ],
        #                     [0,   0,               1.]])
        # return(c_sb @ c_bk)

    # def forward(self, x1, x2, diag=False, **kwargs):
    #     c_b = self.get_c_b()
    #     if diag:
    #         # d = self.calc_hamming_distance(x1, x2, diag=True)
    #         # kernel = c_b[0] + c_b[1] * d + c_b[2] * d ** 2
    #         s = inner_product(x1, x2, diag=True)
    #         kernel = c_b[0] + c_b[1] * s + c_b[2] * torch.square(s)
    #     else:
    #         # calc_d = HammingDistanceCalculator(self.seq_length, scale=c_b[1], shift=c_b[0])
    #         # kernel = calc_d(x1, x2)
    #         ones1 = torch.ones((x1.shape[0], 1), device=x1.device)
    #         ones2 = torch.ones((1, x2.shape[0]), device=x1.device)
    #         s0 = MatmulLinearOperator(c_b[0] * ones1, ones2)
    #         s1 = MatmulLinearOperator(c_b[1] * x1, x2.T)
    #         s2 = SquaredMatMulOperator(c_b[2] * x1, x2.T)
    #         kernel = s0 + s1 + s2
    #     return(kernel)


class VarianceComponentKernel(BaseVarianceComponentKernel):
    is_stationary = True

    def __init__(
        self,
        n_alleles,
        seq_length,
        log_lambdas0=None,
        max_k=None,
        cov0=None,
        ns0=None,
        **kwargs,
    ):
        self.max_k = max_k if max_k is not None else seq_length
        super().__init__(
            n_alleles,
            seq_length,
            log_lambdas0=log_lambdas0,
            cov0=cov0,
            ns0=ns0,
            **kwargs,
        )

    def get_basis(self, d, keops=False):
        for i in range(self.seq_length + 1):
            d_i = float(i)
            b_i = (-self.theta[0, 0] * (d - d_i).abs()).exp()
            yield (b_i)

    def calc_c_bk(self):
        return self.ws.c_bk


class SiteProductKernel(SequenceKernel):
    is_stationary = True

    def __init__(
        self,
        n_alleles,
        seq_length,
        log_var0=None,
        theta0=None,
        cov0=None,
        ns0=None,
        **kwargs,
    ):
        super().__init__(n_alleles, seq_length, **kwargs)
        self.theta0 = theta0
        self.log_var0 = log_var0
        self.set_cov0(cov0, ns0)
        self.set_params()
        self.site_shape = (self.n_alleles, self.n_alleles)

    def calc_log_var0(self):
        if self.log_var0 is None:
            log_var0 = torch.zeros(1)
        else:
            log_var0 = self.log_var0
        return log_var0

    def set_cov0(self, cov0=None, ns0=None):
        self.cov0 = cov0
        self.ns0 = ns0
        self.corr1 = None

        if cov0 is not None:
            if ns0 is None:
                msg = "ns0 must be provided together with cov0"
                raise ValueError(msg)

            y = torch.log(cov0)
            A = torch.stack(
                [torch.ones(self.seq_length + 1), torch.arange(self.seq_length + 1)],
                dim=0,
            ).T
            D = torch.diag(ns0)
            ATD = A.T @ D
            x = torch.linalg.solve(ATD @ A, ATD @ y)
            if x[1].item() < 0:
                self.log_var0 = x[0]
                self.corr1 = torch.exp(x[1])

    def set_params(self):
        theta = Parameter(self.calc_theta0(), requires_grad=True)
        log_var0 = Parameter(self.calc_log_var0(), requires_grad=True)
        self.register_parameter(name="theta", parameter=theta)
        self.register_parameter(name="log_var", parameter=log_var0)

    def _nonkeops_forward(self, x1, x2, diag=False, **kwargs):
        site_kernels = self.get_site_kernels()
        sigma2 = torch.exp(self.log_var)

        x1_ = x1.reshape(x1.shape[0], self.seq_length, self.n_alleles)
        x2_ = x2.reshape(x2.shape[0], self.seq_length, self.n_alleles)

        if diag:
            min_size = min(x1.shape[0], x2.shape[0])
            kernel = torch.einsum(
                "ila,ilb,lab->il", x1_[:min_size], x2_[:min_size], site_kernels
            ).prod(-1)
        else:
            kernel = torch.einsum("ila,jlb,lab->ijl", x1_, x2_, site_kernels).prod(-1)

        return sigma2 * kernel

    def _covar_func(self, x1, x2, **kwargs):
        x1_ = LazyTensor(self.select_site(x1, site=0)[:, None, :])
        x2_ = LazyTensor(self.select_site(x2, site=0)[None, :, :])
        K = (x1_ * x2_).sum(-1)

        for i in range(1, self.seq_length):
            x1_ = LazyTensor(self.select_site(x1, site=i)[:, None, :])
            x2_ = LazyTensor(self.select_site(x2, site=i)[None, :, :])
            K *= (x1_ * x2_).sum(-1)

        return K

    def _keops_forward(self, x1, x2, **kwargs):
        site_kernels = [x for x in self.get_site_kernels()]
        sigma2 = torch.exp(self.log_var)
        x1_ = x1 @ torch.block_diag(*site_kernels)
        kernel = KernelLinearOperator(x1_, x2, covar_func=self._covar_func, **kwargs)
        return sigma2 * kernel

    def get_delta(self):
        delta = self.theta_to_delta(self.theta, n_alleles=self.n_alleles)
        return(delta)

    def get_mutation_delta(self):
        Ks = self.get_site_kernels()
        return(1 - Ks)
    

class ExponentialKernel(SiteProductKernel):
    def get_site_kernel(self):
        rho = torch.exp(self.theta)
        v = (1 - rho) / (1 + (self.n_alleles - 1) * rho)
        kernel = (
            torch.ones((self.n_alleles, self.n_alleles), device=self.theta.device) * v
        ).fill_diagonal_(1.0)
        return kernel

    def calc_theta0(self):
        if self.theta0 is None:
            q = torch.Tensor([0.8]) if self.corr1 is None else self.corr1
            rho = (1 - q) / (1 + (self.n_alleles - 1) * q)
            theta0 = torch.log(rho) * torch.ones(1)
        else:
            theta0 = self.theta0
        return theta0

    def get_site_kernels(self):
        kernel = self.get_site_kernel()
        kernels = torch.stack([kernel] * self.seq_length, axis=0)
        return kernels
    
    def theta_to_delta(self, theta, n_alleles):
        rho = torch.exp(theta)
        delta = 1 - (1 - rho) / (1 + (n_alleles - 1) * rho)
        return(delta)


class ConnectednessKernel(SiteProductKernel):
    def calc_theta0(self):
        if self.theta0 is None:
            if self.corr1 is None:
                theta0 = -3.0 * torch.ones(self.seq_length)
            else:
                rho = (1 - self.corr1) / (1 + (self.n_alleles - 1) * self.corr1)
                theta0 = torch.log(rho) * torch.ones(self.seq_length)
        else:
            theta0 = self.theta0
        return theta0

    def get_site_kernels(self):
        rho = torch.exp(self.theta).reshape((self.seq_length, 1, 1))
        v = (1 - rho) / (1 + (self.n_alleles - 1) * rho)
        kernels = (
            torch.ones(
                (self.seq_length, self.n_alleles, self.n_alleles),
                device=self.theta.device,
            )
            * v
        )
        for i in range(self.seq_length):
            kernels[i].fill_diagonal_(1.0)
        return kernels
    
    def theta_to_delta(self, theta, n_alleles):
        rho = torch.exp(theta)
        delta = 1 - (1 - rho) / (1 + (n_alleles - 1) * rho)
        return(delta)
    

class JengaKernel(SiteProductKernel):
    def calc_theta0(self):
        if self.theta0 is None:
            ones = torch.ones((self.seq_length, self.n_alleles + 1))
            if self.corr1 is None:
                value = -3.0
            else:
                value = torch.log(
                    (1 - self.corr1) / (1 + (self.n_alleles - 1) * self.corr1)
                )
            theta0 = value * ones
        else:
            theta0 = self.theta0
        return theta0

    def get_rho_and_site_factor(self, theta):
        seq_length = theta.shape[0]
        log_rho = theta[:, 0]
        log_p = theta[:, 1:] - torch.logsumexp(theta[:, 1:], dim=1).unsqueeze(
            1
        )
        log_eta = log1mexp(log_p) - log_p
        log_one_p_eta_rho = torch.logaddexp(
            torch.zeros_like(log_eta), log_rho.unsqueeze(1) + log_eta
        )
        site_factors = torch.exp(-0.5 * log_one_p_eta_rho)
        rho = torch.exp(log_rho).reshape((seq_length, 1, 1))
        return(rho, site_factors)

    def get_site_kernels(self):
        rho, site_factors = self.get_rho_and_site_factor(self.theta)
        kernel = (1 - rho) * site_factors.unsqueeze(1) * site_factors.unsqueeze(2)
        for i in range(self.seq_length):
            kernel[i].fill_diagonal_(1.0)
        return kernel
    
    def theta_to_delta(self, theta, **kwargs):
        rho, site_factors = self.get_rho_and_site_factor(theta)
        delta = 1 - torch.sign(1 - rho) * torch.sqrt(torch.abs(1 - rho)) * site_factors
        return(delta)
    

class GeneralProductKernel(SiteProductKernel):
    is_stationary = True

    def __init__(self, n_alleles, seq_length, theta0=None, **kwargs):
        self.dim = int(comb(n_alleles, 2))
        self.theta_to_L = CorrCholeskyTransform()
        super().__init__(n_alleles, seq_length, theta0=theta0, **kwargs)

    def calc_theta0(self):
        if self.theta0 is not None:
            theta0 = self.theta0
        else:
            q = 0.8 if self.corr1 is None else self.corr1
            C = (1 - q) * torch.eye(self.n_alleles) + q * torch.ones(
                (self.n_alleles, self.n_alleles)
            )
            theta0 = self.theta_to_L._inverse(torch.linalg.cholesky(C))
            theta0 = torch.stack([theta0] * self.seq_length, axis=0)
        return theta0
    
    def theta_to_cor(self, theta):
        seq_length = theta.shape[0]
        Ls = [self.theta_to_L(theta[i]) for i in range(seq_length)]
        return torch.stack([(L @ L.T) for L in Ls], axis=0)

    def get_site_kernels(self):
        return self.theta_to_cor(self.theta)
    
    def theta_to_delta(self, theta, **kwargs):
        return(1 - self.theta_to_cor(theta))
    
    def get_delta(self):
        return self.get_mutation_delta()
    

def get_kernel(kernel, n_alleles, seq_length, cov0=None, ns0=None,
               log_var0=None):
    kernels = {
        "Additive": AdditiveKernel,
        "Pairwise": PairwiseKernel,
        "VC": VarianceComponentKernel,
        "Exponential": ExponentialKernel,
        "Connectedness": ConnectednessKernel,
        "Jenga": JengaKernel,
        "GeneralProduct": GeneralProductKernel,
    }
    kernel = kernels[kernel](
        n_alleles, seq_length, cov0=cov0, ns0=ns0, log_var0=log_var0
    )
    return kernel


# class SiteKernel(SequenceKernel):
#     def __init__(self, n_alleles, site, fixed_theta=False, **kwargs):
#         self.site = site
#         self.site_dims = torch.arange(site * n_alleles, (site + 1) * n_alleles)
#         self.fixed_theta = fixed_theta
#         super().__init__(n_alleles, seq_length=1, **kwargs)
#         self.set_params()

#     def select_site(self, x):
#         return(x.index_select(-1, self.site_dims.to(device=x.device)))

#     def calc_theta0(self):
#         if self.theta0 is not None:
#             theta0 = self.theta0
#         else:
#             theta0 = torch.full((self.theta_dim,), fill_value=self.theta_init)
#         return theta0

#     def set_params(self):
#         if self.fixed_theta:
#             self.theta = self.calc_theta0()
#         else:
#             theta0 = self.calc_theta0()
#             params = {"theta": Parameter(theta0, requires_grad=True)}
#             self.register_params(params)

#     def _nonkeops_forward(self, x1, x2, diag=False, **kwargs):
#         x1_, x2_ = self.select_site(x1), self.select_site(x2)
#         site_kernel = self.get_site_kernel()

#         if diag:
#             min_size = min(x1.shape[0], x2.shape[0])
#             kernel = torch.einsum("ia,ib,ab->i", x1_[:min_size], x2_[:min_size], site_kernel)
#         else:
#             kernel = torch.einsum("ia,jb,ab->ij", x1_, x2_, site_kernel)
#         return kernel

#     def _covar_func(self, x1, x2, **kwargs):
#         x1_ = LazyTensor(x1[..., :, None, :])
#         x2_ = LazyTensor(x2[..., None, :, :])
#         K = (x1_ * x2_).sum(-1)
#         return K

#     def _keops_forward(self, x1, x2, **kwargs):
#         site_kernel = self.get_site_kernel()
#         x1_ = self.select_site(x1) @ site_kernel
#         x2_ = self.select_site(x2)
#         kernel = KernelLinearOperator(x1_, x2_, covar_func=self._covar_func, **kwargs)
#         return kernel

#     def get_site_kernel(self):
#         return self.calc_site_kernel(self.theta)


# class ConnectednessSiteKernel(SiteKernel):
#     is_stationary = True
#     def __init__(self, n_alleles, site, theta0=None, **kwargs):
#         self.theta0 = theta0
#         self.theta_init = -3.
#         self.theta_dim = 1
#         self.site_shape = (n_alleles, n_alleles)
#         super().__init__(n_alleles, site, **kwargs)


#     def calc_site_kernel(self, theta):
#         rho = torch.exp(theta)
#         v = (1 - rho) / (1 + (self.n_alleles - 1) * rho)
#         kernel = (torch.ones(self.site_shape, device=theta.device) * v).fill_diagonal_(1.0)
#         return(kernel)


# class JengaSiteKernel(SiteKernel):
#     is_stationary = True
#     def __init__(self, n_alleles, site, theta0=None, **kwargs):
#         self.theta0 = theta0
#         self.theta_init = -3.
#         self.theta_dim = n_alleles + 1
#         super().__init__(n_alleles, site, **kwargs)

#     def calc_site_kernel(self, theta):
#         log_rho = theta[0].item()
#         rho = torch.exp(theta[0])
#         log_p = theta[1:] - torch.logsumexp(theta[1:], dim=0)
#         log_eta = log1mexp(log_p) - log_p
#         log_one_p_eta_rho = torch.logaddexp(torch.zeros_like(log_eta), log_rho + log_eta)
#         site_factors = torch.exp(-0.5 * log_one_p_eta_rho)
#         kernel = ((1 - rho) * site_factors.unsqueeze(0) * site_factors.unsqueeze(1)).fill_diagonal_(1.)
#         return(kernel)


# class GeneralSiteKernel(SiteKernel):
#     is_stationary = True
#     def __init__(self, n_alleles, site, theta0=None, **kwargs):
#         self.theta0 = theta0
#         self.theta_init = -1.0
#         self.theta_dim = int(comb(n_alleles, 2))
#         super().__init__(n_alleles, site, **kwargs)
#         self.theta_to_L = CorrCholeskyTransform()

#     def calc_site_kernel(self, theta):
#         L = self.theta_to_L(theta)
#         site_kernel = L @ L.T
#         return site_kernel


# class SiteProductKernel(ProductKernel):
#     def __init__(self, n_alleles, seq_length, site_kernel, **kwargs):
#         kernels = [site_kernel(n_alleles, site, **kwargs) for site in range(seq_length)]
#         super().__init__(*kernels)


# class ConnectednessKernel(SiteProductKernel):
#     def __init__(self, n_alleles, seq_length, **kwargs):
#         super().__init__(n_alleles, seq_length,
#                          site_kernel=ConnectednessSiteKernel, **kwargs)


# class JengaKernel(SiteProductKernel):
#     def __init__(self, n_alleles, seq_length, **kwargs):
#         super().__init__(n_alleles, seq_length,
#                          site_kernel=JengaSiteKernel, **kwargs)


# class GeneralProductKernel(SiteProductKernel):
#     def __init__(self, n_alleles, seq_length, **kwargs):
#         super().__init__(n_alleles, seq_length,
#                          site_kernel=GeneralSiteKernel, **kwargs)


# class RhoPiKernel(SequenceKernel):
#     def __init__(self, n_alleles, seq_length,
#                  logit_rho0=None, log_p0=None, log_var0=None,
#                  train_p=True, train_var=False,
#                  common_rho=False, correlation=False,
#                  random_init=False,
#                  **kwargs):
#         super().__init__(n_alleles, seq_length, **kwargs)
#         self.logit_rho0 = logit_rho0
#         self.random_init = random_init
#         self.log_p0 = log_p0
#         self.log_var0 = log_var0
#         self.train_p = train_p
#         self.train_var = train_var
#         self.correlation = correlation
#         self.common_rho = common_rho
#         self.set_params()

#     def get_log_p0(self):
#         if self.log_p0 is None:
#             log_p0 = -torch.ones((self.seq_length, self.n_alleles), dtype=self.dtype)
#         else:
#             log_p0 = self.log_p0
#         return(log_p0)

#     def get_logit_rho0(self):
#         # Choose rho0 so that correlation at l/2 is 0.1
#         if self.logit_rho0 is None:
#             shape = (1, 1) if self.common_rho else (self.seq_length, 1)
#             t = np.exp(-2 / self.seq_length * np.log(10.))
#             v = np.log((1 - t) / (self.n_alleles * t))
#             logit_rho0 = torch.full(shape, v, dtype=self.dtype) if self.logit_rho0 is None else self.logit_rho0
#             if self.random_init:
#                 logit_rho0 = torch.normal(logit_rho0, std=1.)
#         else:
#             logit_rho0 = self.logit_rho0
#         return(logit_rho0)

#     def get_log_var0(self, logit_rho0):
#         if self.log_var0 is None:
#             if self.correlation:
#                 rho = torch.exp(logit_rho0) / (1 + torch.exp(logit_rho0))
#                 log_var0 = torch.log(1 + (self.n_alleles - 1) * rho).sum()
#             else:
#                 log_var0 = torch.tensor(0, dtype=self.dtype)
#         else:
#             log_var0 = torch.tensor(self.log_var0, dtype=self.dtype)
#         return(log_var0)

#     def set_params(self):
#         logit_rho0 = self.get_logit_rho0()
#         log_p0 = self.get_log_p0()
#         log_var0 = self.get_log_var0(logit_rho0=logit_rho0)
#         params = {'logit_rho': Parameter(logit_rho0, requires_grad=True),
#                   'log_p': Parameter(log_p0, requires_grad=self.train_p),
#                   'log_var': Parameter(log_var0, requires_grad=self.train_var)}
#         self.register_params(params)

#     def get_log_eta(self):
#         log_p = self.log_p - torch.logsumexp(self.log_p, axis=1).unsqueeze(1)
#         log_eta = log1mexp(log_p) - log_p
#         return(log_eta)

#     def get_log_one_minus_rho(self):
#         return(-torch.logaddexp(self.zeros_like(self.logit_rho), self.logit_rho))

#     def get_factors(self):
#         log1mrho = self.get_log_one_minus_rho()
#         log_rho = self.logit_rho + log1mrho
#         log_eta = self.get_log_eta()
#         log_one_p_eta_rho = torch.logaddexp(self.zeros_like(log_rho), log_rho + log_eta)
#         factors = log_one_p_eta_rho - log1mrho

#         constant = log1mrho.sum()
#         if self.common_rho:
#             constant *= self.seq_length
#         constant += self.log_var
#         return(constant, factors, log_one_p_eta_rho)

#     def _nonkeops_forward(self, x1, x2, diag=False, **params):
#         constant, factors, log_one_p_eta_rho = self.get_factors()
#         factors = factors.reshape(1, self.t)
#         log_one_p_eta_rho = log_one_p_eta_rho.reshape(self.t, 1)

#         if diag:
#             min_size = min(x1.shape[0], x2.shape[0])
#             log_kernel = constant + (x1[:min_size, :] * x2[:min_size, :] * factors).sum(1)
#         else:
#             log_kernel = constant + x1 @ (x2 * factors).T

#         if self.correlation:
#             log_sd1 = 0.5 * (x1 @ log_one_p_eta_rho)
#             log_sd2 = 0.5 * (x2 @ log_one_p_eta_rho)
#             if diag:
#                 log_kernel = log_kernel - log_sd1.flatten() - log_sd2.flatten()
#             else:
#                 log_sd2 = log_sd2.reshape((1, x2.shape[0]))
#                 log_kernel = log_kernel - log_sd1 - log_sd2

#         kernel = torch.exp(log_kernel)
#         return(kernel)

#     def _covar_func(self, x1, x2, constant, **kwargs):
#         x1_ = LazyTensor(x1[..., :, None, :])
#         x2_ = LazyTensor(x2[..., None, :, :])
#         kernel = ((x1_ * x2_).sum(-1) + constant).exp()
#         return(kernel)

#     def _keops_forward(self, x1, x2, **kwargs):
#         # TODO: introduce constants before exponentiation in covar_func
#         constant, factors, log_one_p_eta_rho = self.get_factors()
#         f = factors.reshape(1, self.t)
#         kernel = KernelLinearOperator(x1, x2 * f,
#                                       covar_func=self._covar_func,
#                                       constant=constant, **kwargs)

#         if self.correlation:
#             log_one_p_eta_rho = log_one_p_eta_rho.reshape(1, self.t)
#             sd1_inv_D = DiagLinearOperator(torch.exp(-0.5 * (x1 * log_one_p_eta_rho).sum(1)))
#             sd2_inv_D = DiagLinearOperator(torch.exp(-0.5 * (x2 * log_one_p_eta_rho).sum(1)))
#             kernel = sd1_inv_D @ kernel @ sd2_inv_D

#         return(kernel)


# class ConnectednessKernel(RhoPiKernel):
#     is_stationary = True
#     def __init__(self, n_alleles, seq_length, **kwargs):
#         super().__init__(n_alleles, seq_length,
#                          train_p=False, train_var=False,
#                          correlation=True,
#                          **kwargs)


#     def get_decay_rates(self, positions=None):
#         decay_rates = calc_decay_rates(self.logit_rho.detach().numpy(),
#                                        self.log_p.detach().numpy(),
#                                        sqrt=False, positions=positions).mean(1)
#         return(decay_rates)


# class JengaKernel(RhoPiKernel):
#     is_stationary = True
#     def __init__(self, n_alleles, seq_length, **kwargs):
#         super().__init__(n_alleles, seq_length,
#                          correlation=True, train_p=True, train_var=False,
#                          **kwargs)

#     def get_decay_rates(self, alleles=None, positions=None):
#         decay_rates = calc_decay_rates(self.logit_rho.detach().numpy(),
#                                        self.log_p.detach().numpy(),
#                                        sqrt=True, alleles=alleles, positions=positions)
#         return(decay_rates)


# class ExponentialKernel(ConnectednessKernel):
#     def __init__(self, n_alleles, seq_length, **kwargs):
#         super().__init__(n_alleles, seq_length, common_rho=True, **kwargs)

#     def get_decay_rate(self):
#         decay_rate = calc_decay_rates(self.logit_rho.detach().numpy(),
#                                       self.log_p.detach().numpy(),
#                                       sqrt=False).values.mean()
#         return(decay_rate)


# class GeneralProductKernel_old(SequenceKernel):
#     is_stationary = True
#     def __init__(self, n_alleles, seq_length, theta0=None, **kwargs):
#         super().__init__(n_alleles, seq_length, **kwargs)
#         self.dim = int(comb(n_alleles, 2))
#         self.theta0 = theta0
#         self.set_params()
#         self.theta_to_L = CorrCholeskyTransform()

#     def calc_theta0(self):
#         if self.theta0 is not None:
#             theta0 = self.theta0
#         else:
#             theta0 = torch.zeros((self.seq_length, self.dim), dtype=self.dtype)
#         return(theta0)

#     def theta_to_covs(self, theta):
#         Ls = [self.theta_to_L(theta[i]) for i in range(self.seq_length)]
#         covs = torch.stack([(L @ L.T) for L in Ls], axis=0)
#         return(covs)

#     def set_params(self):
#         theta0 = self.calc_theta0()
#         params = {'theta': Parameter(theta0, requires_grad=True)}
#         self.register_params(params)

#     def get_covs(self):
#         return(self.theta_to_covs(self.theta))

#     def _nonkeops_forward(self, x1, x2, diag=False, **kwargs):
#         covs = self.get_covs()
#         K = self.inner_product(x1[:, :self.n_alleles], x2[:, :self.n_alleles],
#                                metric=covs[0, :, :], diag=diag)
#         for i in range(1, self.seq_length):
#             start, end = i * self.n_alleles, (i+1) * self.n_alleles
#             K *= x1[:, start:end] @ covs[i, :, :] @ x2[:, start:end].T
#         return(K)


# class GeneralProductKernel2(SequenceKernel):
#     is_stationary = True

#     def __init__(self, n_alleles, seq_length, theta0=None, **kwargs):
#         super().__init__(n_alleles, seq_length, **kwargs)
#         self.dim = int(comb(n_alleles, 2))
#         self.theta0 = theta0
#         self.set_params()

#     def calc_theta0(self):
#         if self.theta0 is not None:
#             theta0 = self.theta0
#         else:
#             theta0 = torch.zeros((self.seq_length, self.dim), dtype=self.dtype)
#         return theta0

#     def set_params(self):
#         theta0 = self.calc_theta0()
#         tril_indices = torch.tril_indices(row=self.n_alleles, col=self.n_alleles,
#                                           dtype=torch.int, offset=-1)
#         params = {"theta": Parameter(theta0, requires_grad=True),
#                   'idx': Parameter(tril_indices, requires_grad=False)}
#         self.register_params(params)

#     def theta_to_log_cov(self, theta):
#         v = -torch.exp(theta)
#         log_cov = torch.zeros((self.n_alleles, self.n_alleles),
#                               dtype=theta.dtype, device=theta.device)
#         log_cov[self.idx[0], self.idx[1]] = v
#         log_cov[self.idx[1], self.idx[0]] = v
#         return(log_cov)

#     def get_A(self):
#         return(torch.block_diag(*[self.theta_to_log_cov(self.theta[i, :])
#                                   for i in range(self.seq_length)]))

#     def _nonkeops_forward(self, x1, x2, diag=False, **kwargs):
#         A = self.get_A()
#         return(torch.exp(self.inner_product(x1, x2, metric=A, diag=diag)))

#     def _covar_func(self, x1, x2, **kwargs):
#         x1_ = LazyTensor(x1[:, None, :])
#         x2_ = LazyTensor(x2[None, :, :])
#         kernel = (x1_ * x2_).sum(-1).exp()
#         return(kernel)

#     def _keops_forward(self, x1, x2, **kwargs):
#         A = self.get_A()
#         kernel = KernelLinearOperator(x1, x2 @ A, covar_func=self._covar_func, **kwargs)
#         return(kernel)


# from gpytorch.lazy import delazify

# class AdditiveHeteroskedasticKernel(SequenceKernel):
#     @property
#     def is_stationary(self) -> bool:
#         return self.base_kernel.is_stationary

#     def __init__( self, base_kernel, n_alleles=None, seq_length=None,
#                   log_ds0=None, a=0.5, **kwargs):
#         if base_kernel.active_dims is not None:
#             kwargs["active_dims"] = base_kernel.active_dims

#         if hasattr(base_kernel, 'alpha'):
#             n_alleles = base_kernel.alpha
#         else:
#             if n_alleles is None:
#                 msg = 'If the base kernel does not have n_alleles attribute, '
#                 msg += 'it should be provided'
#                 raise ValueError(msg)

#         if hasattr(base_kernel, 'l'):
#             seq_length = base_kernel.l
#         else:
#             if seq_length is None:
#                 msg = 'If the base kernel does not have seq_length attribute, '
#                 msg += 'it should be provided'
#                 raise ValueError(msg)

#         super().__init__(n_alleles, seq_length, **kwargs)
#         self.log_ds0 = log_ds0
#         self.a = a
#         self.set_params()
#         self.base_kernel = base_kernel

#     def set_params(self):
#         theta = torch.zeros((self.seq_length, self.n_alleles)) if self.log_ds0 is None else self.log_ds0
#         params = {'theta': Parameter(theta, requires_grad=True),
#                   'theta0': Parameter(5 * torch.ones((1,)), requires_grad=True)}
#         self.register_params(params)

#     def get_theta(self):
#         t = self.theta
#         return(t - t.mean(1).unsqueeze(1))

#     def get_theta0(self):
#         return(self.theta0)

#     def f(self, x, theta0, theta, a=0, b=1):
#         phi = theta0 + (x * theta.reshape(1, 1, self.seq_length * self.n_alleles)).sum(-1)
#         r = a + (b - a) * torch.exp(phi) / (1 + torch.exp(phi))
#         return(r)

#     def forward(self, x1, x2, last_dim_is_batch=False, diag=False, **params):
#         orig_output = self.base_kernel.forward(x1, x2, diag=diag,
#                                                last_dim_is_batch=last_dim_is_batch,
#                                                **params)
#         theta0, theta = self.get_theta0(), self.get_theta()
#         f1 = self.f(x1, theta0, theta, a=self.a).T
#         f2 = self.f(x2, theta0, theta, a=self.a)

#         if last_dim_is_batch:
#             f1 = f1.unsqueeze(-1)
#             f2 = f2.unsqueeze(-1)
#         if diag:
#             f1 = f1.unsqueeze(-1)
#             f2 = f2.unsqueeze(-1)
#             return(f1 * f2 * delazify(orig_output))
#         else:
#             return(f1 * f2 * orig_output)

#     def num_outputs_per_input(self, x1, x2):
#         return self.base_kernel.num_outputs_per_input(x1, x2)

#     def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
#         return self.base_kernel.prediction_strategy(train_inputs, train_prior_dist, train_labels, likelihood)
