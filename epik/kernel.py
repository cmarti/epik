import sys

import numpy as np
import torch as torch
from gpytorch.kernels import Kernel
from linear_operator.operators import KernelLinearOperator
from pykeops.torch import LazyTensor
from scipy.optimize import minimize
from scipy.special import comb, logsumexp
from torch.distributions.transforms import CorrCholeskyTransform
from torch.distributions import Wishart, Beta
from torch.nn import Parameter
from torch.linalg import cholesky

from epik.utils import (
    HammingDistanceCalculator,
    KrawtchoukPolynomials,
    inner_product,
    log1mexp,
    cov2corr
)


class SequenceKernel(Kernel):
    """
    A kernel class for sequence data, inheriting from the base `Kernel` class.

    This class implements methods for calculating kernel matrices and 
    Hamming distances for sequence data, with optional support for 
    KeOps for efficient computation on large datasets.
    
    Parameters
    ----------
    n_alleles : int
        The number of alleles in the sequence data.
    seq_length : int
        The length of the sequences.
    use_keops : bool, optional
        Whether to use KeOps for kernel computation (default is False).
    **kwargs : dict
        Additional keyword arguments passed to the base `Kernel` class.
    """
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

                except RuntimeError as error:  # Memory error
                    msg = "\n{}. Likely due to memory error when loading ".format(error)
                    msg += "kernel matrix: switching to KeOps\n"
                    sys.stderr.write(msg)

                    allocated_memory = torch.cuda.memory_allocated(device="cuda")
                    reserved_memory = torch.cuda.memory_reserved(device="cuda")
                    n1, n2 = x1.shape[0], x2.shape[0]
                    sys.stderr.write(
                        f"Kernel matrix memory {(n1, n2)}: {n1 * n2 * x2.element_size() / 1e6} MB"
                    )
                    sys.stderr.write(
                        f"Memory allocated: {allocated_memory / 1e6:.2f} MB\n"
                    )
                    sys.stderr.write(
                        f"Memory reserved: {reserved_memory / 1e6:.2f} MB\n"
                    )
                    sys.stderr.write(torch.cuda.memory_summary(device="cuda"))
                    self.use_keops = True
                    torch.cuda.empty_cache()
                    kernel = self._keops_forward(x1, x2, **kwargs)

        torch.cuda.empty_cache()
        return kernel


class BaseVarianceComponentKernel(SequenceKernel):
    is_stationary = True

    def __init__(self, n_alleles, seq_length, log_lambdas0=None, **kwargs):
        super().__init__(n_alleles, seq_length, **kwargs)
        self.log_lambdas0 = log_lambdas0
        self.ws = KrawtchoukPolynomials(n_alleles, seq_length, max_k=self.max_k)
        self.set_params()

    def get_log_lambdas0(self):
        if self.log_lambdas0 is None:
            log_lambdas0 = torch.zeros(self.max_k + 1)
            # else:
            #     b = self.cov0[1] / (self.cov0[0] - self.cov0[1])
            #     beta0 = np.log(self.cov0[0]) + self.seq_length * (
            #         np.log(1 + self.n_alleles * b) - np.log(1 + b)
            #     )
            #     beta1 = np.log(1 + self.n_alleles * b)
            #     k = torch.arange(self.ws.max_k + 1).to(dtype=torch.float)
            #     log_lambdas0 = beta0 - beta1 * k
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
    r"""
    Additive kernel for functions on sequence space.

    This kernel computes the covariance between two sequences as a linear function
    of the Hamming distance separating them. The parameters are derived from the
    variance contributions of the constant and additive components.

    .. math::
        K(x, y) = c_0 + c_1 \cdot d(x, y)

    where:
    
    .. math::
        c_0 = \lambda_0 + \ell \cdot (\alpha - 1) \cdot \lambda_1
        
    .. math::
        c_1 = -\alpha \cdot \lambda_1

    Here, :math:`\lambda_0` and :math:`\lambda_1` are variance parameters, :math:`\ell` is the sequence
    length, and :math:`\alpha` is the number of alleles.

    When applied to one-hot encoded sequence embeddings :math:`x_1` and :math:`x_2`, this kernel
    returns a linear operator that facilitates efficient matrix-vector products
    without explicitly constructing the full covariance matrix.
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
    Pairwise kernel for functions on sequence space.
     
    The covariance between two sequences is quadratic in the Hamming distance
    that separates them, with coefficients determined by the variance
    explained by the constant, additive and pairwise components.

    .. math::
        K(x, y) = c_0 + c_1 \cdot d(x, y) + c_2 \cdot d(x, y)^2

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
    """
    Variance Component Kernel for functions on sequence space.

    This kernel computes the covariance between two sequences using
    Krawtchouk polynomials.

    .. math::
        K(x, y) = \sum_{k=0}^{\ell} \lambda_k \cdot K_k(x, y)

    To ensure differentiability in PyTorch, the covariance for each
    distance class is precomputed, and a kernel interpolation approach
    is used to compute the covariance between input sequence pairs.
    """
    is_stationary = True

    def __init__(
        self,
        n_alleles,
        seq_length,
        log_lambdas0=None,
        max_k=None,
        **kwargs,
    ):
        self.max_k = max_k if max_k is not None else seq_length
        super().__init__(
            n_alleles,
            seq_length,
            log_lambdas0=log_lambdas0,
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
        **kwargs,
    ):
        super().__init__(n_alleles, seq_length, **kwargs)
        self.theta0 = theta0
        self.log_var0 = log_var0
        self.set_params()
        self.site_shape = (self.n_alleles, self.n_alleles)

    def is_positive(self):
        return False

    def calc_log_var0(self):
        if self.log_var0 is None:
            log_var0 = torch.zeros(1)
        else:
            log_var0 = self.log_var0
        return log_var0

    def set_params(self):
        theta = Parameter(self.calc_theta0(), requires_grad=True)
        log_var0 = Parameter(self.calc_log_var0(), requires_grad=True)
        self.register_parameter(name="theta", parameter=theta)
        self.register_parameter(name="log_var", parameter=log_var0)

    def _nonkeops_forward(self, x1, x2, diag=False, **kwargs):
        if self.is_positive():
            site_log_kernels = self.get_site_log_kernels()

            if diag:
                min_size = min(x1.shape[0], x2.shape[0])
                log_kernel = 0.0
                for i in range(self.seq_length):
                    log_kernel += (
                        (self.select_site(x1[:min_size], site=i) @ site_log_kernels[i])
                        * self.select_site(x2[:min_size], site=i)
                    ).sum(1)

            else:
                log_kernel = 0
                for i in range(self.seq_length):
                    log_kernel += (
                        self.select_site(x1, site=i)
                        @ site_log_kernels[i]
                        @ self.select_site(x2, site=i).T
                    )

            return torch.exp(self.log_var + log_kernel)

        else:
            site_kernels = self.get_site_kernels()
            sigma2 = torch.exp(self.log_var)

            if diag:
                min_size = min(x1.shape[0], x2.shape[0])
                kernel = 1.0
                for i in range(self.seq_length):
                    kernel *= (
                        (self.select_site(x1, site=i) @ site_kernels[i])
                        * self.select_site(x2, site=i)
                    ).sum(1)

            else:
                kernel = 1.0
                for i in range(self.seq_length):
                    kernel *= (
                        self.select_site(x1, site=i)
                        @ site_kernels[i]
                        @ self.select_site(x2, site=i).T
                    )

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

    def _covar_func_log(self, x1, x2, **kwargs):
        x1_ = LazyTensor(x1[:, None, :])
        x2_ = LazyTensor(x2[None, :, :])
        K = (x1_ * x2_).sum(-1).exp()
        return K

    def _keops_forward(self, x1, x2, **kwargs):
        if self.is_positive():
            site_log_kernels = self.get_site_log_kernels()
            M = torch.block_diag(*site_log_kernels)
            sigma2 = torch.exp(self.log_var)
            kernel = sigma2 * KernelLinearOperator(
                x1 @ M, x2, covar_func=self._covar_func_log, **kwargs
            )

        else:
            site_kernels = [x for x in self.get_site_kernels()]
            sigma2 = torch.exp(self.log_var)
            M = torch.block_diag(*site_kernels)
            kernel = sigma2 * KernelLinearOperator(
                x1 @ M, x2, covar_func=self._covar_func, **kwargs
            )
        return kernel

    def get_delta(self):
        """
        Compute the decay factors of the kernel.

        The decay factors represent the percentage decrease in predictability 
        when introducing a specific mutation.

        Returns
        -------
        delta : torch.Tensor
            A tensor containing the decay factors.
        """
        delta = self.theta_to_delta(self.theta, n_alleles=self.n_alleles)
        return delta

    def get_mutation_delta(self):
        """
        Compute the mutation-specific decay factors of the kernel.

        The decay factors represent the percentage decrease in predictability 
        when introducing a specific mutation.

        Returns
        -------
        delta : torch.Tensor
            A tensor containing the decay factors for each possible mutation.
        """
        Ks = self.get_site_kernels()
        return 1 - Ks


class ExponentialKernel(SiteProductKernel):
    r"""
    Exponential Kernel for functions on sequence space.

    This kernel computes the covariance between two sequences as a
    geometrically decaying function of the Hamming distance separating them.


    .. math::
        K(x, y) = \left( \frac{ 1-\rho }{ 1 + (\alpha - 1)\rho } \right)^d

    where:
    
        - :math:`\rho` is a parameter controlling the decay rate.
        
        - :math:`\alpha` is the number of alleles.
        
        - :math:`d` is the Hamming distance between sequences :math:`x` and :math:`y`.

    """
    def get_site_kernel(self):
        rho = torch.exp(self.theta)
        v = (1 - rho) / (1 + (self.n_alleles - 1) * rho)
        kernel = (
            torch.ones((self.n_alleles, self.n_alleles), device=self.theta.device) * v
        ).fill_diagonal_(1.0)
        return kernel

    def get_site_log_kernel(self):
        w = log1mexp(self.theta) - torch.log1p(
            (self.n_alleles - 1) * torch.exp(self.theta)
        )
        log_kernel = (
            torch.ones((self.n_alleles, self.n_alleles), device=self.theta.device) * w
        ).fill_diagonal_(0.0)
        return log_kernel

    def calc_theta0(self):
        if self.theta0 is None:
            q = torch.Tensor([np.exp(-np.log(10) / self.seq_length)])
            qs = Beta(20 * q, 20 * (1 - q)).sample((1, ))
            rho = (1 - qs) / (1 + (self.n_alleles - 1) * qs)
            theta0 = torch.log(rho)
        else:
            theta0 = self.theta0
        return theta0

    def get_site_kernels(self):
        kernel = self.get_site_kernel()
        kernels = torch.stack([kernel] * self.seq_length, axis=0)
        return kernels

    def get_site_log_kernels(self):
        log_kernel = self.get_site_log_kernel()
        log_kernels = torch.stack([log_kernel] * self.seq_length, axis=0)
        return log_kernels

    def theta_to_delta(self, theta, n_alleles):
        rho = torch.exp(theta)
        delta = 1 - (1 - rho) / (1 + (n_alleles - 1) * rho)
        return delta

    def is_positive(self):
        return torch.all(self.theta < 0.0)

    def _nonkeops_forward(self, x1, x2, diag=False, **kwargs):
        if self.is_positive():
            w = log1mexp(self.theta) - torch.log1p(
                (self.n_alleles - 1) * torch.exp(self.theta)
            )
            d = self.calc_hamming_distance(x1, x2, diag=diag, keops=False)
            return torch.exp(self.log_var + w * d)

        else:
            return super()._nonkeops_forward(x1, x2, diag=diag, **kwargs)


class ConnectednessKernel(SiteProductKernel):
    r"""
    Connectedness Kernel for functions on sequence space.

    This kernel computes the covariance between two sequences where
    mutations at different sites have different effects on the 
    predictability of other mutations
    

    .. math::
        K(x, y) = \prod_p^{\ell}\frac{1-\rho_p}{1 + (\alpha - 1)\rho_p}

    where:
    
        - :math:`\rho_p` is a parameter controlling the decay rate of site :math:`p`.
        
        - :math:`\alpha` is the number of alleles.
        
        - :math:`\ell` is the sequence length.

    """
    def calc_theta0(self):
        if self.theta0 is None:
            q = torch.Tensor([np.exp(-np.log(10) / self.seq_length)])
            qs = Beta(20 * q, 20 * (1 - q)).sample((self.seq_length, ))
            rho = (1 - qs) / (1 + (self.n_alleles - 1) * qs)
            theta0 = torch.log(rho)
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

    def get_site_log_kernels(self):
        ws = log1mexp(self.theta) - torch.log1p(
            (self.n_alleles - 1) * torch.exp(self.theta)
        )
        log_kernels = []
        size = (self.n_alleles, self.n_alleles)
        for w in ws:
            log_kernel = w * torch.ones(size, device=self.theta.device)
            log_kernels.append(log_kernel.fill_diagonal_(0.0))
        return torch.stack(log_kernels, axis=0)

    def is_positive(self):
        return torch.all(self.theta < 0.0)

    def theta_to_delta(self, theta, n_alleles):
        rho = torch.exp(theta)
        delta = 1 - (1 - rho) / (1 + (n_alleles - 1) * rho)
        return delta


class JengaKernel(SiteProductKernel):
    r"""
    Jenga Kernel for functions on sequence space.

    This kernel computes the covariance between two sequences as the product 
    of allele- and site-specific factors at the alleles where they differ.

    .. math::
        K(x, y) = \prod_{p: x_p \neq y_p} 
        \sqrt{\frac{1-\rho_p}{1 + \frac{1-\pi_p^{x_p}}{\pi_p^{x_p}}\rho_p}}
        \sqrt{\frac{1-\rho_p}{1 + \frac{1-\pi_p^{y_p}}{\pi_p^{y_p}}\rho_p}}

    where:
    
        - :math:`\rho_p` is a parameter controlling the decay rate at site :math:`p`.
        
        - :math:`\pi_p^{x_p}` and :math:`\pi_p^{y_p}` are site and allele specific probabilities.
        
        - :math:`\ell` is the sequence length.
    """

    def calc_theta0(self):
        if self.theta0 is None:
            q = torch.Tensor([np.exp(-np.log(10) / self.seq_length)])
            qs = Beta(20 * q, 20 * (1 - q)).sample((self.seq_length, ))
            rho = (1 - qs) / (1 + (self.n_alleles - 1) * qs)
            theta0 = torch.randn((self.seq_length, self.n_alleles + 1))
            theta0[:, :1] = torch.log(rho)
        else:
            theta0 = self.theta0
        return theta0

    def get_log_rho_log1p_eta_rho(self, theta):
        log_rho = theta[:, 0].unsqueeze(1)
        log_p = theta[:, 1:] - torch.logsumexp(theta[:, 1:], 1).unsqueeze(1)
        log_eta = log1mexp(log_p) - log_p

        log1p_eta_rho = torch.logaddexp(torch.zeros_like(log_eta), log_rho + log_eta)
        return log_rho, log1p_eta_rho

    def get_site_kernels(self):
        log_rho, log1p_eta_rho = self.get_log_rho_log1p_eta_rho(self.theta)
        rho = torch.exp(log_rho).reshape((self.seq_length, 1, 1))
        allele_factors = 0.5 * log1p_eta_rho
        denom = torch.exp(-allele_factors.unsqueeze(1) - allele_factors.unsqueeze(2))
        kernel = (1 - rho) * denom
        for i in range(self.seq_length):
            kernel[i].fill_diagonal_(1.0)
        return kernel

    def get_site_log_kernels(self):
        log_rho, log1p_eta_rho = self.get_log_rho_log1p_eta_rho(self.theta)
        log1m_rho = log1mexp(log_rho)
        zs = 0.5 * (log1m_rho - log1p_eta_rho)

        log_kernels = []
        for z in zs:
            log_kernel = z.unsqueeze(0) + z.unsqueeze(1)
            log_kernels.append(log_kernel.fill_diagonal_(0.0))
        return torch.stack(log_kernels, axis=0)

    def is_positive(self):
        return torch.all(self.theta[:, 0] < 0.0)

    def theta_to_delta(self, theta, **kwargs):
        log_rho, log1p_eta_rho = self.get_log_rho_log1p_eta_rho(theta)
        onemrho = 1 - torch.exp(log_rho)
        site_factors = torch.exp(-0.5 * log1p_eta_rho)
        delta = 1 - torch.sign(onemrho) * torch.sqrt(torch.abs(onemrho)) * site_factors
        return delta


class GeneralProductKernel(SiteProductKernel):
    """
    General Product Kernel for sequence data.

    This kernel computes the covariance between two sequences as the product
    of site-specific kernels, where each site kernel is parameterized by the
    Cholesky factor of a correlation matrix.

    .. math::
        K(x, y) = \prod_{p=1}^\ell K_p(x_p, y_p),

    where:
    
    .. math::
        K_p = L L^T,

    and :math:`L` is the Cholesky factor of the correlation matrix, 
    parameterized using the LKJ transform.
    """
    is_stationary = True

    def __init__(self, n_alleles, seq_length, theta0=None, **kwargs):
        self.dim = int(comb(n_alleles, 2))
        self.theta_to_L = CorrCholeskyTransform()
        super().__init__(n_alleles, seq_length, theta0=theta0, **kwargs)

    def calc_theta0(self):
        if self.theta0 is not None:
            theta0 = self.theta0
        else:
            q = torch.Tensor([np.exp(-np.log(10) / self.seq_length)])
            C = (1 - q) * torch.eye(self.n_alleles) + q * torch.ones(
                (self.n_alleles, self.n_alleles)
            )
            df = 21.
            Ls = [cholesky(cov2corr(Wishart(torch.Tensor([df]), C).sample()[0]))
                  for _ in range(self.seq_length)]
            theta0 = torch.stack([self.theta_to_L._inverse(L)
                                  for L in Ls], axis=0)
        return theta0

    def theta_to_cor(self, theta):
        seq_length = theta.shape[0]
        Ls = [self.theta_to_L(theta[i]) for i in range(seq_length)]
        return torch.stack([(L @ L.T) for L in Ls], axis=0)

    def get_site_kernels(self):
        return self.theta_to_cor(self.theta)
    
    def get_site_log_kernels(self):
        return torch.log(self.get_site_kernels())

    def is_positive(self):
        return torch.all(self.get_site_kernels() > 0.0)

    def theta_to_delta(self, theta, **kwargs):
        return 1 - self.theta_to_cor(theta)

    def get_delta(self):
        return self.get_mutation_delta()


def get_kernel(
    kernel, n_alleles, seq_length, theta0=None, log_var0=None, log_lambdas0=None
):
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
        n_alleles,
        seq_length,
        theta0=theta0,
        log_var0=log_var0,
        log_lambdas0=log_lambdas0,
    )
    return kernel


class SiteKernelAligner(object):
    def __init__(self, n_alleles, seq_length):
        self.n_alleles = n_alleles
        self.seq_length = seq_length
        self.size = (n_alleles, n_alleles)
        self.n_offdiag = n_alleles**2 - n_alleles
        self.theta_to_L = CorrCholeskyTransform()

    def calc_frob(self, A, B):
        return np.square(A - B).sum()

    def log_rho_to_q(self, log_rho):
        rho = np.exp(log_rho)
        v = (1 - rho) / (1 + (self.n_alleles - 1) * rho)
        return v

    def q_to_log_rho(self, q):
        log_rho = np.log((1 - q) / (1 + (self.n_alleles - 1) * q))
        return log_rho

    def rho_to_corr(self, theta):
        v = self.log_rho_to_q(theta)
        corr = np.full(self.size, v)
        np.fill_diagonal(corr, 1)
        return corr

    def corr_to_q(self, corr):
        q = (corr.sum() - np.diag(corr).sum()) / self.n_offdiag
        return q

    def corr_to_general_product(self, corr):
        L = torch.Tensor(np.linalg.cholesky(corr))
        return self.theta_to_L._inverse(L).numpy()

    def general_product_to_corr(self, theta):
        L = self.theta_to_L(torch.Tensor(theta))
        return (L @ L.T).numpy()

    def exponential_to_connectedness(self, theta):
        return np.vstack([theta] * self.seq_length)

    def exponential_to_jenga(self, theta):
        col1 = np.vstack([theta] * self.seq_length)
        return np.hstack([col1, np.zeros((self.seq_length, self.n_alleles))])

    def exponential_to_general_product(self, theta):
        corr = self.rho_to_corr(theta[0])
        theta = np.vstack([self.corr_to_general_product(corr)] * self.seq_length)
        return theta

    def connectedness_to_exponential(self, theta):
        q = np.mean([self.log_rho_to_q(theta_i) for theta_i in theta])
        theta = np.array([[self.q_to_log_rho(q)]])
        return theta

    def connectedness_to_jenga(self, theta):
        z = np.zeros((self.seq_length, self.n_alleles))
        return np.hstack([theta, z])

    def connectedness_to_general_product(self, theta):
        theta = np.vstack(
            [
                self.corr_to_general_product(self.rho_to_corr(theta_i))
                for theta_i in theta
            ]
        )
        return theta

    def jenga_to_corr(self, theta):
        log_rho, log_p = theta[0], theta[1:]
        rho = np.exp(log_rho)
        p = np.exp(log_p - logsumexp(log_p))
        eta = (1 - p) / p
        fs = np.sqrt(1 + eta * rho)
        corr = (1 - rho.reshape((1, 1))) / (
            np.expand_dims(fs, 0) * np.expand_dims(fs, 1)
        )
        np.fill_diagonal(corr, 1)
        return corr

    def jenga_to_connectedness(self, theta):
        qs = [self.corr_to_q(self.jenga_to_corr(theta_i)) for theta_i in theta]
        theta = np.array([[self.q_to_log_rho(q)] for q in qs])
        return theta

    def jenga_to_exponential(self, theta):
        q = np.mean([self.corr_to_q(self.jenga_to_corr(theta_i)) for theta_i in theta])
        theta = np.array([[self.q_to_log_rho(q)]])
        return theta

    def jenga_to_general_product(self, theta):
        theta = np.vstack(
            [
                self.corr_to_general_product(self.jenga_to_corr(theta_i))
                for theta_i in theta
            ]
        )
        return theta

    def general_product_to_exponential(self, theta):
        q = np.mean(
            [self.corr_to_q(self.general_product_to_corr(theta_i)) for theta_i in theta]
        )
        theta = np.array([[self.q_to_log_rho(q)]])
        return theta

    def general_product_to_connectedness(self, theta):
        qs = [
            self.corr_to_q(self.general_product_to_corr(theta_i)) for theta_i in theta
        ]
        theta = np.array([[self.q_to_log_rho(q)] for q in qs])
        return theta

    def general_product_to_jenga(self, theta):
        thetas = []
        for theta_i in theta:
            B = self.general_product_to_corr(theta_i)

            def loss(params):
                A = self.jenga_to_corr(params)
                return self.calc_frob(A, B)

            params0 = np.zeros(self.n_alleles + 1)
            res = minimize(loss, x0=params0)
            thetas.append(res.x)
        return np.vstack(thetas)


class FactorAnalysisKernel(Kernel):
    def __init__(self, n_alleles, seq_length, ndim, **kwargs):
        self.n_alleles = n_alleles
        self.seq_length = seq_length
        self.ndim = ndim
        self.nfeatures = seq_length * n_alleles
        super().__init__(**kwargs)

        q_raw0 = torch.normal(0, 1, size=(self.nfeatures, self.ndim))
        q_raw = Parameter(q_raw0, requires_grad=True)

        log_lambdas0_sqrt0 = torch.linspace(0, -2, self.ndim).unsqueeze(0)
        log_lambdas0_sqrt = Parameter(log_lambdas0_sqrt0, requires_grad=True)
        log_sigma2 = Parameter(torch.Tensor([0.]), requires_grad=True)

        self.register_parameter(name="log_lambdas_sqrt", parameter=log_lambdas0_sqrt)
        self.register_parameter(name="q_raw", parameter=q_raw)
        self.register_parameter(name="log_sigma2", parameter=log_sigma2)

    @property
    def lambdas_sqrt(self):
        return(torch.exp(self.log_lambdas_sqrt))

    @property
    def q(self):
        return(torch.linalg.qr(self.q_raw)[0])
    
    @property
    def sigma2(self):
        return torch.exp(self.log_sigma2)

    def forward(self, x1, x2, diag=False, **kwargs):
        Q = self.q * self.lambdas_sqrt
        v1 = x1 @ Q
        v2 = x2 @ Q

        if diag:
            min_size = min(x1.shape[0], x2.shape[0])
            kernel = self.sigma2 * torch.ones((min_size, ))
            
        else:
            # Equivalent to squared distance in the linear subspace
            z1 = torch.sum(torch.square(v1), axis=1).unsqueeze(1)
            z2 = torch.sum(torch.square(v2), axis=1).unsqueeze(0)
            kernel = torch.exp(self.log_sigma2 + 2 * v1 @ v2.T - z1 - z2)

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
