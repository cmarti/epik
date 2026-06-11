import sys
from itertools import product
from typing import Any, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from gpytorch.kernels import Kernel
from linear_operator.operators import (
    BlockDiagLinearOperator,
    KernelLinearOperator,
    LinearOperator,
    to_linear_operator,
)
from pykeops.torch import LazyTensor
from scipy.optimize import minimize
from scipy.special import comb, logsumexp
from torch.distributions.transforms import (
    CorrCholeskyTransform,
    LowerCholeskyTransform,
)
from torch.linalg import cholesky
from torch.nn import Parameter
from torch.utils.checkpoint import checkpoint

from epik.utils import (
    HammingDistanceCalculator,
    KrawtchoukPolynomials,
    diff_inner_prod,
    inner_product,
    log1mexp,
    validate_alphabet,
)


class SequenceKernel(Kernel):
    """
    A kernel class for sequence data, inheriting from the base `Kernel` class.

    This class implements methods for calculating kernel matrices and
    Hamming distances for sequence data, with optional support for
    KeOps for efficient computation on large datasets.

    Parameters
    ----------
    seq_length : int
        The length of the sequences.
    alphabet_type: str, optional
        Type of alphabet used ('rna', 'dna', 'protein'). Default is None.
    alphabet : list[str], optional
        List of characters specifying the alphabet for all sites.
    alphabet_list : list[list[str]], optional
        Per-site alphabets; the list length must match `seq_length`.
    use_keops : bool, optional
        Whether to use KeOps for kernel computation (default is False).
    **kwargs : dict
        Additional keyword arguments passed to the base `Kernel` class.
    """

    def __init__(
        self,
        seq_length: Optional[int] = None,
        alphabet_type: Optional[str] = None,
        alphabet: Optional[List[str]] = None,
        alphabet_list: Optional[List[List[str]]] = None,
        use_keops: bool = False,
        **kwargs: Any,
    ) -> None:
        self.alphabet_list = validate_alphabet(
            seq_length, alphabet_type, alphabet, alphabet_list
        )
        self.alphas = np.array([len(site) for site in self.alphabet_list])
        self.alphabet_full = np.unique(np.concatenate(self.alphabet_list))
        self.alpha = self.alphabet_full.shape[0]

        self.starts = np.cumsum(np.append([0], self.alphas[:-1]))
        self.ends = np.cumsum(self.alphas)
        self.alleles_pos = {
            a: [
                i
                for i, alphabet in enumerate(self.alphabet_list)
                if a in alphabet
            ]
            for a in self.alphabet_full
        }
        self.alleles_idx = {
            a: [
                self.starts[i] + self.alphabet_list[i].index(a)
                for i in allele_pos
            ]
            for a, allele_pos in self.alleles_pos.items()
        }

        self.l = len(self.alphabet_list)
        self.lp1 = self.l + 1
        self.positions = list(range(self.l))

        self.n_features = np.sum(self.alphas)

        self.logn = np.sum(np.log(self.alphas))
        self.logam1 = np.log(self.alphas - 1)
        self.logam1_max = np.max(self.logam1)
        self.loga = np.log(self.alphas)
        self.loga_max = np.max(self.loga)
        self.a_max = np.max(self.alphas)

        self.use_keops = use_keops
        super().__init__(**kwargs)

    def select_site(self, x: torch.Tensor, site: int) -> torch.Tensor:
        if not (0 <= site < self.l):
            raise IndexError("Site index out of range")
        return x[..., self.starts[site] : self.ends[site]]

    def select_allele(self, x: torch.Tensor, allele: str) -> torch.Tensor:
        if allele not in self.alleles_idx:
            raise ValueError(f"Allele '{allele}' not found in the alphabet.")
        indices = self.alleles_idx[allele]
        return x[..., indices]

    def check_input_shape(self, x: torch.Tensor) -> None:
        if x.shape[1] != self.n_features:
            msg = f"Input tensor must have {self.n_features} features, but got {x.shape[1]}."
            raise ValueError(msg)

    def calc_hamming_distance(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        keops: bool = False,
    ) -> Union[torch.Tensor, LazyTensor]:
        """
        Calculate the Hamming distance between two sets of sequences.

        Parameters
        ----------
        x1 : torch.Tensor
            Tensor of one-hot encoded sequences.
        x2 : torch.Tensor
            Tensor of one-hot encoded sequences.
        diag : bool, optional
            If True, compute only the diagonal of the Hamming distance matrix. Default is False.
        keops : bool, optional
            If True, use KeOps for efficient computation on large datasets. Default is False.

        Returns
        -------
        torch.Tensor or LazyTensor
            The computed Hamming distance matrix or its diagonal.
        """
        self.check_input_shape(x1)
        self.check_input_shape(x2)

        if diag or not keops:
            s = inner_product(x1, x2, diag=diag)
            d = float(self.l) - s
        else:
            x1_ = LazyTensor(x1.contiguous()[..., :, None, :])
            x2_ = LazyTensor(x2.contiguous()[..., None, :, :])
            s = (x1_ * x2_).sum(-1)
            d = float(self.l) - s
        return d

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        **kwargs: Any,
    ) -> Union[torch.Tensor, KernelLinearOperator]:
        """
        Compute the kernel matrix or its diagonal between two input tensors.
        Parameters
        ----------
        x1 : torch.Tensor
            Tensor of one-hot encoded sequences.
        x2 : torch.Tensor
            Tensor of one-hot encoded sequences.
        diag : bool, optional
            If True, compute only the diagonal of the kernel matrix. Default is False.
        **kwargs : Any
            Additional arguments for kernel computation.
        Returns
        -------
        torch.Tensor or KernelLinearOperator
            The computed kernel matrix or its diagonal.
        """

        self.check_input_shape(x1)
        self.check_input_shape(x2)

        if diag:
            kernel = self._nonkeops_forward(x1, x2, diag=True, **kwargs)

        elif self.use_keops:
            kernel = self._keops_forward(x1, x2, **kwargs)

        else:
            try:
                kernel = self._nonkeops_forward(x1, x2, diag=False, **kwargs)

            except RuntimeError as error:  # Memory error
                msg = f"\n{error}. Likely due to memory error when loading "
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

    def save(self, fpath):
        """
        Save the kernel parameters to a file for future use.

        Parameters
        ----------
        fpath : str
            The file path where the model parameters will be saved.
        """
        torch.save(self.state_dict(), fpath)

    def load(self, fpath, **kwargs):
        """
        Load kernel parameters from a file.

        Parameters
        ----------
        fpath : str
            Path to the file containing the stored model parameters.

        **kwargs : dict, optional
            Additional arguments to pass to `torch.load` for loading the parameters.
        """
        params = torch.load(fpath, **kwargs)
        self.load_state_dict(params)


class CorrelationKernel(SequenceKernel):
    def get_log_var0(self, log_var0: Optional[torch.Tensor]):
        if log_var0 is None:
            log_var0 = torch.zeros((1,))
        if log_var0.shape != (1,):
            raise ValueError("log_var0 should be a tensor of shape (1,)")

        return log_var0


class VarianceComponentKernel(SequenceKernel):
    r"""
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
        seq_length: Optional[int] = None,
        alphabet_type: Optional[str] = None,
        alphabet: Optional[List[str]] = None,
        alphabet_list: Optional[List[List[str]]] = None,
        log_lambdas0: Optional[torch.Tensor] = None,
        max_k: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            seq_length=seq_length,
            alphabet_type=alphabet_type,
            alphabet=alphabet,
            alphabet_list=alphabet_list,
            **kwargs,
        )
        self.max_alpha = int(max(self.alphas))
        self.max_k = max_k if max_k is not None else self.l
        self.set_log_lambdas0(log_lambdas0)
        self.ws = KrawtchoukPolynomials(
            self.max_alpha, self.l, max_k=self.max_k
        )
        self.set_params()

    def set_log_lambdas0(self, log_lambdas0: Optional[torch.Tensor]) -> None:
        if log_lambdas0 is None:
            log_lambdas0 = torch.zeros(self.max_k + 1)
        elif log_lambdas0.shape[0] != self.max_k + 1:
            msg = f"`log_lambdas0` must have length {self.max_k + 1}, "
            msg += f"but got length {log_lambdas0.shape[0]}."
            raise ValueError(msg)

        self.log_lambdas0 = log_lambdas0.to(dtype=self.dtype)

    def set_params(self) -> None:
        c_bk = Parameter(self.calc_c_bk(), requires_grad=False)
        theta = Parameter(
            torch.Tensor([[-np.log(0.1) / self.l]]), requires_grad=False
        )
        log_lambdas = Parameter(self.log_lambdas0, requires_grad=True)

        self.register_parameter(name="c_bk", parameter=c_bk)
        self.register_parameter(name="theta", parameter=theta)
        self.register_parameter(name="log_lambdas", parameter=log_lambdas)

    def calc_c_b(self, log_lambdas: torch.Tensor) -> torch.Tensor:
        c_b = self.c_bk @ torch.exp(log_lambdas)
        return c_b

    def get_c_b(self) -> torch.Tensor:
        return self.calc_c_b(self.log_lambdas)

    def distance_to_cov(
        self,
        d: Union[torch.Tensor, LazyTensor],
        c_b: torch.Tensor,
        keops: bool = False,
    ) -> Union[torch.Tensor, LazyTensor]:
        basis = self.calc_basis(d, keops=keops)
        b_0 = next(basis)
        c_0 = c_b[0]

        kernel = c_0 * b_0
        for b_i, c_i in zip(basis, c_b[1:]):
            kernel += c_i * b_i
        return kernel

    def _compute(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        log_lambdas: torch.Tensor,
        diag: bool = False,
        **kwargs: Any,
    ) -> Union[torch.Tensor, LazyTensor, KernelLinearOperator]:
        c_b = self.calc_c_b(log_lambdas)
        d = self.calc_hamming_distance(x1, x2, diag=diag)
        return self.distance_to_cov(d, c_b, keops=False)

    def _nonkeops_forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        **kwargs: Any,
    ) -> Union[torch.Tensor, KernelLinearOperator]:
        return checkpoint(
            self._compute,
            x1,
            x2,
            self.log_lambdas,
            diag,
            preserve_rng_state=True,
        )

    def _covar_func(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        c_b: torch.Tensor,
        **kwargs: Any,
    ) -> LazyTensor:
        d = self.calc_hamming_distance(x1, x2, keops=True)
        return self.distance_to_cov(d, c_b, keops=True)

    def _keops_forward(
        self, x1: torch.Tensor, x2: torch.Tensor, **kwargs: Any
    ) -> KernelLinearOperator:
        c_b = self.get_c_b()
        kernel = KernelLinearOperator(
            x1, x2, covar_func=self._covar_func, c_b=c_b, **kwargs
        )
        return kernel

    def calc_basis(
        self, d: Union[torch.Tensor, LazyTensor], keops: bool = False
    ) -> Generator[torch.Tensor, None, None]:
        for i in range(self.l + 1):
            d_i = float(i)
            b_i = (-self.theta[0, 0] * (d - d_i).abs()).exp()
            yield (b_i)

    def calc_c_bk(self) -> torch.Tensor:
        return self.ws.c_bk


class AdditiveKernel(VarianceComponentKernel):
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

    def __init__(
        self,
        seq_length: Optional[int] = None,
        alphabet_type: Optional[str] = None,
        alphabet: Optional[List[str]] = None,
        alphabet_list: Optional[List[List[str]]] = None,
        log_lambdas0: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            seq_length=seq_length,
            alphabet_type=alphabet_type,
            alphabet=alphabet,
            alphabet_list=alphabet_list,
            log_lambdas0=log_lambdas0,
            max_k=1,
            **kwargs,
        )

    def calc_c_bk(self) -> torch.Tensor:
        a, sl = self.max_alpha, self.l
        c_bk = torch.tensor([[1.0, sl * (a - 1)], [0, -a]])
        return c_bk

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        **kwargs: Any,
    ) -> Union[torch.Tensor, KernelLinearOperator]:
        self.check_input_shape(x1)
        self.check_input_shape(x2)

        c_b = self.get_c_b()
        if diag:
            d = self.calc_hamming_distance(x1, x2, diag=True)
            kernel = c_b[0] + c_b[1] * d
        else:
            calc_d = HammingDistanceCalculator(
                self.l, scale=c_b[1], shift=c_b[0]
            )
            kernel = calc_d(x1, x2)
        return kernel


class PairwiseKernel(VarianceComponentKernel):
    r"""
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

    def __init__(
        self,
        seq_length: Optional[int] = None,
        alphabet_type: Optional[str] = None,
        alphabet: Optional[List[str]] = None,
        alphabet_list: Optional[List[List[str]]] = None,
        log_lambdas0: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            seq_length=seq_length,
            alphabet_type=alphabet_type,
            alphabet=alphabet,
            alphabet_list=alphabet_list,
            log_lambdas0=log_lambdas0,
            max_k=2,
            **kwargs,
        )

    def calc_basis(
        self, d: torch.Tensor, keops: bool = False
    ) -> Generator[torch.Tensor, None, None]:
        b0 = 1.0 if keops else torch.ones_like(d)
        yield (b0)
        yield (d)
        yield (d.square())

    def calc_c_bk(self) -> torch.Tensor:
        a, sl = self.max_alpha, self.l
        c13 = (
            a * sl
            - 0.5 * a**2 * sl
            - 0.5 * sl
            - a * sl**2
            + 0.5 * a**2 * sl**2
            + 0.5 * sl**2
        )
        c23 = -a + 0.5 * a**2 + a * sl - a**2 * sl
        c_bk = torch.tensor(
            [[1, sl * (a - 1), c13], [0, -a, c23], [0, 0, 0.5 * a**2]]
        )
        return c_bk


class SiteProductKernel(CorrelationKernel):
    is_stationary = True

    def get_corr1d0(self) -> torch.Tensor:
        return np.exp(-np.log(10) / self.l)

    def get_log_mu0(self, log_mu0: Optional[torch.Tensor]) -> torch.Tensor:
        if log_mu0 is None:
            q = self.get_corr1d0()
            qs = torch.Tensor([q] * self.n_mu)
            mu = (1 - qs) / (1 + (self.a_max - 1) * qs)  # type: ignore
            log_mu0 = torch.log(mu)

        if log_mu0.shape != (self.n_mu,):
            raise ValueError(
                f"theta0 shape should be ({self.l},) but got {log_mu0.shape}"
            )

        return log_mu0

    def _nonkeops_forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        **kwargs: Any,
    ) -> Union[torch.Tensor, KernelLinearOperator]:
        if self.is_positive():
            site_log_kernels = self.get_site_log_kernels()

            if diag:
                min_size = min(x1.shape[0], x2.shape[0])
                log_kernel = 0.0
                for i in range(self.l):
                    log_kernel += (
                        (
                            self.select_site(x1[:min_size], site=i)
                            @ site_log_kernels[i]
                        )
                        * self.select_site(x2[:min_size], site=i)
                    ).sum(1)

            else:
                log_kernel = 0
                for i in range(self.l):
                    log_kernel += (
                        self.select_site(x1, site=i)
                        @ site_log_kernels[i]
                        @ self.select_site(x2, site=i).T
                    )

            return torch.exp(self.log_var + log_kernel)

        else:
            site_kernels = self.get_site_kernels()
            kernel = torch.exp(self.log_var).item()

            if diag:
                min_size = min(x1.shape[0], x2.shape[0])
                for i in range(self.l):
                    x1i = self.select_site(x1, site=i)
                    x2i = self.select_site(x2, site=i)
                    kernel *= ((x1i @ site_kernels[i]) * x2i).sum(1)

            else:
                for i in range(self.l):
                    x1i = self.select_site(x1, site=i)
                    x2i = self.select_site(x2, site=i)
                    kernel *= x1i @ site_kernels[i] @ x2i.T

            return kernel

    def _covar_func(
        self, x1: torch.Tensor, x2: torch.Tensor, **kwargs: Any
    ) -> LazyTensor:
        K = 1.0
        for i in range(self.l):
            x1_ = LazyTensor(
                self.select_site(x1, site=i).contiguous()[:, None, :]
            )
            x2_ = LazyTensor(
                self.select_site(x2, site=i).contiguous()[None, :, :]
            )
            K *= (x1_ * x2_).sum(-1)

        return K

    def _covar_func_log(
        self, x1: torch.Tensor, x2: torch.Tensor, **kwargs: Any
    ) -> LazyTensor:
        x1_ = LazyTensor(x1.contiguous()[:, None, :])
        x2_ = LazyTensor(x2.contiguous()[None, :, :])
        K = (x1_ * x2_).sum(-1).exp()
        return K

    def _keops_forward(
        self, x1: torch.Tensor, x2: torch.Tensor, **kwargs: Any
    ) -> KernelLinearOperator:
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

    def get_mutation_delta(self) -> List[torch.Tensor]:
        """
        Compute the mutation-specific decay factors of the kernel.

        The decay factors represent the percentage decrease in predictability
        when introducing a specific mutation.

        Returns
        -------
        delta : list of torch.Tensor
            A list of tensors containing the decay factors for each possible mutation
            at each site.
        """
        return [1 - K for K in self.get_site_kernels()]


class AlleleSymmetricProductKernel(SiteProductKernel):
    def _set_params(
        self,
        log_var0: Optional[torch.Tensor] = None,
        log_mu0: Optional[torch.Tensor] = None,
    ) -> None:
        log_mu = Parameter(self.get_log_mu0(log_mu0), requires_grad=True)
        log_var = Parameter(self.get_log_var0(log_var0), requires_grad=True)

        self.register_parameter(name="log_mu", parameter=log_mu)
        self.register_parameter(name="log_var", parameter=log_var)

    def is_positive(self) -> bool:
        return torch.all(self.log_mu < 0.0)

    def log_mu_to_log_delta(self, log_mu: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(log_mu)
        log1p_eta_mu = torch.logaddexp(zeros, self.logam1_max + log_mu)
        log_delta = log_mu + self.loga_max - log1p_eta_mu
        return log_delta

    def log_mu_to_log_corr_1d(self, log_mu: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(log_mu)
        log1p_eta_mu = torch.logaddexp(zeros, self.logam1_max + log_mu)
        return log1mexp(log_mu) - log1p_eta_mu

    def log_mu_to_delta(self, log_mu: torch.Tensor) -> torch.Tensor:
        log_delta = self.log_mu_to_log_delta(log_mu)
        return torch.exp(log_delta)

    def get_delta(self) -> torch.Tensor:
        """
        Compute the decay factors of the kernel.

        The decay factors represent the percentage decrease in predictability
        when introducing a specific mutation.

        Returns
        -------
        delta : torch.Tensor
            A tensor containing the decay factors.
        """
        return self.log_mu_to_delta(self.log_mu)


class GeometricKernel(AlleleSymmetricProductKernel):
    r"""
    Geometric Kernel for functions on sequence space.

    This kernel computes the covariance between two sequences as a
    geometrically decaying function of the Hamming distance separating them.


    .. math::
        K(x, y) = \sigma^2 \left( \frac{ 1-\mu }{ 1 + (\alpha - 1)\mu } \right)^d

    where:

        - :math:`\sigma^2` corresponds to the kernel variance.

        - :math:`\mu` is a parameter controlling the decay rate.

        - :math:`\alpha` is the max number of alleles across sites.

        - :math:`d` is the Hamming distance between sequences :math:`x` and :math:`y`.

    """

    def __init__(
        self,
        seq_length: Optional[int] = None,
        alphabet_type: Optional[str] = None,
        alphabet: Optional[List[str]] = None,
        alphabet_list: Optional[List[List[str]]] = None,
        log_var0: Optional[torch.Tensor] = None,
        log_mu0: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> None:
        SequenceKernel.__init__(
            self,
            seq_length=seq_length,
            alphabet_type=alphabet_type,
            alphabet=alphabet,
            alphabet_list=alphabet_list,
            **kwargs,
        )
        self.n_mu = 1
        self.site_shapes = [(a, a) for a in self.alphas]
        self._set_params(log_var0, log_mu0)

    def get_site_kernels(self) -> List[torch.Tensor]:
        v = -torch.expm1(self.log_mu_to_log_delta(self.log_mu))

        kernels = []
        for shape in self.site_shapes:
            kernel = v * torch.ones(shape, device=self.log_mu.device)  # type: ignore
            kernels.append(kernel.fill_diagonal_(1.0))

        return kernels

    def get_site_log_kernels(self) -> List[torch.Tensor]:
        w = self.log_mu_to_log_corr_1d(self.log_mu)  # type: ignore

        log_kernels = []
        for shape in self.site_shapes:
            log_kernel = w * torch.ones(shape, device=self.log_mu.device)  # type: ignore
            log_kernels.append(log_kernel.fill_diagonal_(0.0))

        return log_kernels

    def _nonkeops_forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        **kwargs: Any,
    ) -> Union[torch.Tensor, KernelLinearOperator]:
        d = self.calc_hamming_distance(x1, x2, diag=diag, keops=self.use_keops)
        if self.is_positive():
            w = self.log_mu_to_log_corr_1d(self.log_mu)  # type: ignore
            return (self.log_var + w * d).exp()

        else:
            v = 1 - self.log_mu_to_delta(self.log_mu)  # type: ignore
            return torch.exp(self.log_var) * v**d

    def _covar_func(
        self, x1: torch.Tensor, x2: torch.Tensor, **kwargs: Any
    ) -> LazyTensor:
        K = 1.0
        for i in range(self.l):
            x1_ = LazyTensor(
                self.select_site(x1, site=i).contiguous()[:, None, :]
            )
            x2_ = LazyTensor(
                self.select_site(x2, site=i).contiguous()[None, :, :]
            )
            K *= (x1_ * x2_).sum(-1)

        return K

    def _keops_forward(
        self, x1: torch.Tensor, x2: torch.Tensor, **kwargs: Any
    ) -> KernelLinearOperator:
        if self.is_positive():
            w = self.log_mu_to_log_corr_1d(self.log_mu)
            kernel = KernelLinearOperator(
                self.log_var + w - w * x1,
                x2,
                covar_func=self._covar_func_log,
                **kwargs,
            )

        else:
            site_kernels = [x for x in self.get_site_kernels()]
            sigma2 = torch.exp(self.log_var)
            M = torch.block_diag(*site_kernels)
            kernel = sigma2 * KernelLinearOperator(
                x1 @ M, x2, covar_func=self._covar_func, **kwargs
            )
        return kernel


class ConnectednessKernel(AlleleSymmetricProductKernel):
    r"""
    Connectedness Kernel for functions on sequence space.

    This kernel computes the covariance between two sequences where
    mutations at different sites have different effects on the
    predictability of other mutations


    .. math::
        K(x, y) = \sigma^2 \prod_p^{\ell}\frac{1-\mu_p}{1 + (\alpha - 1)\mu_p}

    where:

        - :math:`\sigma^2` corresponds to the kernel variance.

        - :math:`\mu_p` is a parameter controlling the decay rate of site :math:`p`.

        - :math:`\alpha` is the max number of alleles across sites.

        - :math:`\ell` is the sequence length.

    """

    def __init__(
        self,
        seq_length: Optional[int] = None,
        alphabet_type: Optional[str] = None,
        alphabet: Optional[List[str]] = None,
        alphabet_list: Optional[List[List[str]]] = None,
        log_var0: Optional[torch.Tensor] = None,
        log_mu0: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> None:
        SequenceKernel.__init__(
            self,
            seq_length=seq_length,
            alphabet_type=alphabet_type,
            alphabet=alphabet,
            alphabet_list=alphabet_list,
            **kwargs,
        )
        self.n_mu = self.l
        self.site_shapes = [(a, a) for a in self.alphas]
        self._set_params(log_var0, log_mu0)

    def get_site_kernels(self) -> List[torch.Tensor]:
        vs = -torch.expm1(self.log_mu_to_log_delta(self.log_mu))

        kernels = []
        for v, shape in zip(vs, self.site_shapes):
            kernel = v * torch.ones(shape, device=self.log_mu.device)  # type: ignore
            kernels.append(kernel.fill_diagonal_(1.0))

        return kernels

    def get_site_log_kernels(self) -> torch.Tensor:
        ws = self.log_mu_to_log_corr_1d(self.log_mu)  # type: ignore

        log_kernels = []
        for w, shape in zip(ws, self.site_shapes):
            log_kernel = w * torch.ones(shape, device=self.log_mu.device)  # type: ignore
            log_kernels.append(log_kernel.fill_diagonal_(0.0))

        return log_kernels


class JengaKernel(SiteProductKernel):
    r"""
    Jenga Kernel for functions on sequence space.

    This kernel computes the covariance between two sequences as the product
    of allele- and site-specific factors at the alleles where they differ.

    .. math::
        K(x, y) = \sigma^2 \prod_{p: x_p \neq y_p}
        \sqrt{\frac{1-\mu_p}{1 + \frac{1-\pi_p^{x_p}}{\pi_p^{x_p}}\mu_p}}
        \sqrt{\frac{1-\mu_p}{1 + \frac{1-\pi_p^{y_p}}{\pi_p^{y_p}}\mu_p}}

    where:

        - :math:`\sigma^2` corresponds to the kernel variance.

        - :math:`\mu_p` is a parameter controlling the decay rate at site :math:`p`.

        - :math:`\pi_p^{x_p}` and :math:`\pi_p^{y_p}` are site and allele specific probabilities.

        - :math:`\ell` is the sequence length.
    """

    def __init__(
        self,
        seq_length: Optional[int] = None,
        alphabet_type: Optional[str] = None,
        alphabet: Optional[List[str]] = None,
        alphabet_list: Optional[List[List[str]]] = None,
        log_var0: Optional[torch.Tensor] = None,
        log_mu0: Optional[torch.Tensor] = None,
        log_pi0: Optional[List[torch.Tensor]] = None,
        **kwargs: Any,
    ) -> None:
        SequenceKernel.__init__(
            self,
            seq_length=seq_length,
            alphabet_type=alphabet_type,
            alphabet=alphabet,
            alphabet_list=alphabet_list,
            **kwargs,
        )
        self.n_mu = self.l
        self.site_shapes = [(a, a) for a in self.alphas]
        self._set_params(log_var0, log_mu0, log_pi0)

    def get_log_pi0(
        self, log_pi0: Optional[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        if log_pi0 is None:
            log_pi0 = [torch.zeros(alpha) for alpha in self.alphas]

        if len(log_pi0) != self.l:
            raise ValueError(
                f"log_pi0 should have dimension {self.l}, but got {len(log_pi0)}"
            )

        shapes = [x.shape for x in log_pi0]
        exp_shapes = [(alpha,) for alpha in self.alphas]

        if shapes != exp_shapes:
            raise ValueError(
                f"log_pi0 shapes should be ({exp_shapes},) but got {shapes}"
            )
        return log_pi0

    def _set_params(
        self,
        log_var0: Optional[torch.Tensor],
        log_mu0: Optional[torch.Tensor],
        log_pi0: Optional[List[torch.Tensor]],
    ) -> None:
        log_var = Parameter(self.get_log_var0(log_var0), requires_grad=True)
        self.register_parameter(name="log_var", parameter=log_var)

        log_mu = Parameter(self.get_log_mu0(log_mu0), requires_grad=True)
        self.register_parameter(name="log_mu", parameter=log_mu)

        for i, log_pi0_i in enumerate(self.get_log_pi0(log_pi0)):
            log_pi0_i = Parameter(log_pi0_i, requires_grad=True)
            self.register_parameter(name=f"log_pi_{i}", parameter=log_pi0_i)

    def is_positive(self) -> bool:
        return torch.all(self.log_mu < 0.0)  # type: ignore

    def get_log_pi_p(self, p):
        log_pi_p = getattr(self, f"log_pi_{p}")
        log_pi_p = log_pi_p - torch.logsumexp(log_pi_p, dim=0)
        return log_pi_p

    def get_log1p_eta_mu_p(self, p) -> List[torch.Tensor]:
        log_pi_p = self.get_log_pi_p(p)
        log_eta_p = log1mexp(log_pi_p) - log_pi_p
        log1p_eta_mu_p = torch.logaddexp(
            torch.zeros_like(log_eta_p), self.log_mu[p] + log_eta_p
        )
        return log1p_eta_mu_p

    def get_site_kernels(self) -> List[torch.Tensor]:
        kernels = []
        for p in range(self.l):
            kernel = -torch.expm1(self.get_log_delta_p(p)).fill_diagonal_(-1.0)
            kernels.append(kernel)
        return kernels

    def get_log_corr_p_a(self, p: int) -> torch.Tensor:
        return 0.5 * (log1mexp(self.log_mu[p]) - self.get_log1p_eta_mu_p(p))

    def get_site_log_kernels(self) -> List[torch.Tensor]:
        log_kernels = []
        for p in range(self.l):
            v = self.get_log_corr_p_a(p)
            log_kernel = (v.unsqueeze(0) + v.unsqueeze(1)).fill_diagonal_(0.0)
            log_kernels.append(log_kernel)

        return log_kernels

    def get_log_delta_p(self, p):
        log_mu_p = self.log_mu[p]
        v = 0.5 * self.get_log1p_eta_mu_p(p)
        m = v.unsqueeze(0) + v.unsqueeze(1)
        log_mu_p_m = torch.logaddexp(log_mu_p, m)
        log_delta_p = log1mexp(-log_mu_p_m) - m + log_mu_p_m
        return log_delta_p

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
        return [torch.exp(self.get_log_delta_p(p)) for p in range(self.l)]


class GeneralProductKernel(SiteProductKernel):
    r"""
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

    def __init__(
        self,
        seq_length: Optional[int] = None,
        alphabet_type: Optional[str] = None,
        alphabet: Optional[List[str]] = None,
        alphabet_list: Optional[List[List[str]]] = None,
        log_var0: Optional[torch.Tensor] = None,
        theta0: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> None:
        SequenceKernel.__init__(
            self,
            seq_length=seq_length,
            alphabet_type=alphabet_type,
            alphabet=alphabet,
            alphabet_list=alphabet_list,
            **kwargs,
        )
        self.site_shapes = [(a, a) for a in self.alphas]
        self.dims = [int(comb(a, 2)) for a in self.alphas]
        self.theta_to_L = CorrCholeskyTransform()
        self._set_params(
            log_var0=log_var0,
            theta0=theta0,
        )

    def _set_params(
        self,
        log_var0: Optional[torch.Tensor],
        theta0: Optional[torch.Tensor],
    ) -> None:
        log_var = Parameter(self.get_log_var0(log_var0), requires_grad=True)
        self.register_parameter(name="log_var", parameter=log_var)

        for p, theta_p in enumerate(self.get_theta0(theta0)):
            theta_p = Parameter(theta_p, requires_grad=True)
            self.register_parameter(name=f"theta_{p}", parameter=theta_p)

    def get_theta0(self, theta0) -> List[torch.Tensor]:
        if theta0 is None:
            q = self.get_corr1d0()
            theta0 = []
            for a in self.alphas:
                C = (1 - q) * torch.eye(a) + q * torch.ones((a, a))
                v = self.cor_to_theta(C)
                theta0.append(v)

        if len(theta0) != self.l:
            raise ValueError(
                f"theta0 should have length {self.l} but got {len(theta0)}"
            )

        return theta0

    def get_theta_p(self, p):
        return getattr(self, f"theta_{p}")

    def cor_to_theta(self, C):
        return self.theta_to_L._inverse(cholesky(C))

    def theta_to_cor(self, theta: torch.Tensor) -> torch.Tensor:
        L = self.theta_to_L(theta)
        return L @ L.T

    def get_site_kernels(self) -> List[torch.Tensor]:
        kernels = [
            self.theta_to_cor(self.get_theta_p(p)) for p in range(self.l)
        ]
        return kernels

    def get_site_log_kernels(self) -> List[torch.Tensor]:
        return [torch.log(k) for k in self.get_site_kernels()]

    def is_positive(self) -> bool:
        # return(False)
        for k in self.get_site_kernels():
            if torch.any(k <= 0):
                return False
        return True

    def theta_to_delta(self, theta: torch.Tensor) -> torch.Tensor:
        return 1 - self.theta_to_cor(theta)

    def get_delta(self) -> List[torch.Tensor]:
        return [
            self.theta_to_delta(self.get_theta_p(p)) for p in range(self.l)
        ]


class GeneralProductKernel2(SiteProductKernel):
    r"""
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

    def __init__(
        self,
        seq_length: Optional[int] = None,
        alphabet_type: Optional[str] = None,
        alphabet: Optional[List[str]] = None,
        alphabet_list: Optional[List[List[str]]] = None,
        log_var0: Optional[torch.Tensor] = None,
        theta0: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> None:
        SequenceKernel.__init__(
            self,
            seq_length=seq_length,
            alphabet_type=alphabet_type,
            alphabet=alphabet,
            alphabet_list=alphabet_list,
            **kwargs,
        )
        self.site_shapes = [(a, a) for a in self.alphas]
        self.dims = [int(comb(a, 2)) for a in self.alphas]
        self.theta_to_L = CorrCholeskyTransform()
        self._set_params(
            log_var0=log_var0,
            theta0=theta0,
        )

    def _set_params(
        self,
        log_var0: Optional[torch.Tensor],
        theta0: Optional[torch.Tensor],
    ) -> None:
        log_var = Parameter(self.get_log_var0(log_var0), requires_grad=True)
        self.register_parameter(name="log_var", parameter=log_var)

        for p, theta_p in enumerate(self.get_theta0(theta0)):
            theta_p = Parameter(theta_p, requires_grad=True)
            self.register_parameter(name=f"theta_{p}", parameter=theta_p)

    def get_theta0(self, theta0) -> List[torch.Tensor]:
        if theta0 is None:
            q = self.get_corr1d0()
            theta0 = []
            for a in self.alphas:
                C = (1 - q) * torch.eye(a) + q * torch.ones((a, a))
                v = self.cor_to_theta(C)
                theta0.append(v)

        if len(theta0) != self.l:
            raise ValueError(
                f"theta0 should have length {self.l} but got {len(theta0)}"
            )

        return theta0

    def get_theta_p(self, p):
        return getattr(self, f"theta_{p}")

    def cor_to_theta(self, C):
        return self.theta_to_L._inverse(cholesky(C))

    def get_site_log_kernels(self) -> List[torch.Tensor]:
        log_kernels = []
        for p in self.positions:
            log_kernels.append(self.theta_to_log_cor(self.get_theta_p(p)))
        return log_kernels

    def get_site_kernels(self) -> List[torch.Tensor]:
        kernels = []
        for p in self.positions:
            kernels.append(self.theta_to_cor(self.get_theta_p(p)))
        return kernels

    def theta_to_log_cor(self, theta: torch.Tensor) -> torch.Tensor:
        L_p = self.theta_to_L(theta)
        return -diff_inner_prod(v1=L_p, v2=L_p, diag=False)

    def theta_to_cor(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.theta_to_log_cor(theta))

    def is_positive(self) -> bool:
        return True

    def theta_to_delta(self, theta: torch.Tensor) -> torch.Tensor:
        return 1 - self.theta_to_cor(theta)

    def get_delta(self) -> List[torch.Tensor]:
        return [
            self.theta_to_delta(self.get_theta_p(p)) for p in range(self.l)
        ]


class DiploidKernel(CorrelationKernel):
    def __init__(
        self,
        seq_length: Optional[int] = None,
        alphabet_type: Optional[str] = None,
        alphabet: Optional[List[str]] = None,
        alphabet_list: Optional[List[List[str]]] = None,
        log_var0: Optional[torch.Tensor] = None,
        log_lambda0: Optional[torch.Tensor] = None,
        log_eta0: Optional[torch.Tensor] = None,
        logit_p0: Optional[torch.Tensor] = None,
        partition_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        SequenceKernel.__init__(
            self,
            seq_length=seq_length,
            alphabet_type=alphabet_type,
            alphabet=alphabet,
            alphabet_list=alphabet_list,
            **kwargs,
        )
        self._set_params(log_var0, log_lambda0, log_eta0, logit_p0)
        self.partition_size = partition_size
        self.n_features = 3 * self.l
        if self.alpha > 2:
            raise ValueError("DiploidKernel only supports binary alphabets.")

    def _set_params(
        self,
        log_var0: Optional[torch.Tensor],
        log_lambda0: Optional[torch.Tensor],
        log_eta0: Optional[torch.Tensor],
        logit_p0: Optional[torch.Tensor],
    ) -> None:
        log_var = Parameter(self.get_log_var0(log_var0), requires_grad=True)
        self.register_parameter(name="log_var", parameter=log_var)

        log_lambda = Parameter(
            self.get_log_lambda0(log_lambda0), requires_grad=True
        )
        self.register_parameter(name="log_lambda", parameter=log_lambda)

        log_eta = Parameter(self.get_log_eta0(log_eta0), requires_grad=True)
        self.register_parameter(name="log_eta", parameter=log_eta)

        logit_p = Parameter(self.get_logit_p(logit_p0), requires_grad=True)
        self.register_parameter(name="logit_p", parameter=logit_p)

    def get_log_lambda0(
        self, log_lambda0: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if log_lambda0 is None:
            log_lambda0 = torch.tensor([-1.0])

        if log_lambda0.shape != (1,):
            raise ValueError(
                f"log_lambda0 should have shape (1,) but got {log_lambda0}"
            )
        return log_lambda0

    def get_log_eta0(self, log_eta0: Optional[torch.Tensor]) -> torch.Tensor:
        if log_eta0 is None:
            log_eta0 = torch.tensor([-1.0])

        if log_eta0.shape != (1,):
            raise ValueError(
                f"log_eta0 should have shape (1,) but got {log_eta0}"
            )
        return log_eta0

    def get_logit_p(self, logit_p0: Optional[torch.Tensor]) -> torch.Tensor:
        if logit_p0 is None:
            logit_p0 = torch.tensor([0.0])

        if logit_p0.shape != (1,):
            raise ValueError(
                f"logit_p0 should have shape (1,) but got {logit_p0}"
            )
        return logit_p0

    def _expand_param(self, param: torch.Tensor) -> torch.Tensor:
        if param.ndim != 1:
            raise ValueError(
                f"param must be 1D, but got shape {tuple(param.shape)}"
            )
        if param.shape[0] == 1:
            return param.expand(self.l)
        if param.shape[0] == self.l:
            return param
        raise ValueError(
            f"param must have length 1 or {self.l}, but got {param.shape[0]}"
        )

    def get_site_log_kernel(self) -> torch.Tensor:
        logit_p = self._expand_param(self.logit_p)
        log_lambda = self._expand_param(self.log_lambda)
        log_eta = self._expand_param(self.log_eta)

        zeros = torch.zeros_like(logit_p)
        log_1mp = torch.logaddexp(zeros, logit_p)
        log_1p_eta_nu = torch.logaddexp(zeros, log_eta + logit_p)
        log_denom = torch.logaddexp(log_1p_eta_nu, log_1mp + log_lambda)

        log_1mp_lambda = log_1mp + log_lambda

        log_d1 = log1mexp(log_eta) - log_denom
        log_d2 = (
            log_1p_eta_nu
            + log1mexp(log_1mp_lambda - log_1p_eta_nu)
            - log_denom
        )
        log_het = torch.logaddexp(zeros, log_eta - logit_p) - log_denom

        log_kernel = torch.zeros(
            (self.l, 3, 3), dtype=logit_p.dtype, device=logit_p.device
        )
        log_kernel[:, 0, 1] = log_d1
        log_kernel[:, 1, 0] = log_d1
        log_kernel[:, 2, 1] = log_d1
        log_kernel[:, 1, 2] = log_d1
        log_kernel[:, 2, 0] = log_d2
        log_kernel[:, 0, 2] = log_d2
        log_kernel[:, 1, 1] = log_het
        return log_kernel

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, diag=False, **kwargs: Any
    ) -> torch.Tensor:

        blocks = self.get_site_log_kernel()
        blocks += self.log_var / self.l
        M = BlockDiagLinearOperator(to_linear_operator(blocks), block_dim=-3)
        if self.partition_size is None or diag:
            kernel = torch.exp(inner_product(x1, x2, metric=M, diag=diag))
        else:
            x2_T = M @ x2.T
            kernel = PartitionedExpKernelOperator(x1, x2_T, partition_size=self.partition_size)
        return kernel


class PartitionedExpKernelOperator(LinearOperator):
    def __init__(self, x1, x2_T, partition_size: int):
        super().__init__(x1, x2_T, partition_size=partition_size)
        self.partition_size = partition_size
        self.x1 = x1
        self.x2_T = x2_T
        self._shape = torch.Size((x1.shape[0], x2_T.shape[1]))

    def representation(self):
        return self.x1, self.x2_T

    def _size(self):
        return self._shape

    def _transpose_nonbatch(self):
        return PartitionedExpKernelOperator(
            self.x2_T.transpose(-1, -2),
            self.x1.transpose(-1, -2),
            self.partition_size,
        )

    def _matmul(self, rhs):
        out = rhs.new_zeros(self.x1.shape[0], rhs.shape[-1])

        for i in range(0, self.x1.shape[0], self.partition_size):
            e = min(i + self.partition_size, self.x1.shape[0])
            out[i:e] = torch.exp(self.x1[i:e] @ self.x2_T) @ rhs

        return out

def get_kernel(
    kernel: str,
    n_alleles: int,
    seq_length: int,
    theta0: Optional[torch.Tensor] = None,
    log_var0: Optional[torch.Tensor] = None,
    log_lambdas0: Optional[torch.Tensor] = None,
    ndim: int = 3,
) -> SequenceKernel:
    kernels = {
        "Additive": AdditiveKernel,
        "Pairwise": PairwiseKernel,
        "VC": VarianceComponentKernel,
        "Geometric": GeometricKernel,
        "Connectedness": ConnectednessKernel,
        "Jenga": JengaKernel,
        "GeneralProduct": GeneralProductKernel,
        "FactorAnalysis": FactorAnalysisKernel,
        "MahalanobisRBF": MahalanobisRBFKernel,
    }
    kernel = kernels[kernel](
        n_alleles,
        seq_length,
        theta0=theta0,
        log_var0=log_var0,
        log_lambdas0=log_lambdas0,
        ndim=ndim,
    )
    return kernel


class SiteKernelAligner:
    def __init__(self, n_alleles: int, seq_length: int) -> None:
        self.n_alleles = n_alleles
        self.l = seq_length
        self.size = (n_alleles, n_alleles)
        self.n_offdiag = n_alleles**2 - n_alleles
        self.theta_to_L = CorrCholeskyTransform()

    def calc_frob(self, A: np.ndarray, B: np.ndarray) -> float:
        return np.square(A - B).sum()

    def log_mu_to_q(self, log_mu: float) -> float:
        mu = np.exp(log_mu)
        v = (1 - mu) / (1 + (self.n_alleles - 1) * mu)
        return v

    def q_to_log_mu(self, q: float) -> float:
        log_mu = np.log((1 - q) / (1 + (self.n_alleles - 1) * q))
        return log_mu

    def mu_to_corr(self, theta: float) -> np.ndarray:
        v = self.log_mu_to_q(theta)
        corr = np.full(self.size, v)
        np.fill_diagonal(corr, 1)
        return corr

    def corr_to_q(self, corr: np.ndarray) -> float:
        q = (corr.sum() - np.diag(corr).sum()) / self.n_offdiag
        return q

    def corr_to_general_product(self, corr: np.ndarray) -> np.ndarray:
        L = torch.Tensor(np.linalg.cholesky(corr))
        return self.theta_to_L._inverse(L).numpy()

    def general_product_to_corr(self, theta: np.ndarray) -> np.ndarray:
        L = self.theta_to_L(torch.Tensor(theta))
        return (L @ L.T).numpy()

    def geometric_to_connectedness(self, theta: np.ndarray) -> np.ndarray:
        return np.vstack([theta] * self.l)

    def geometric_to_jenga(self, theta: np.ndarray) -> np.ndarray:
        col1 = np.vstack([theta] * self.l)
        return np.hstack([col1, np.zeros((self.l, self.n_alleles))])

    def geometric_to_general_product(self, theta: np.ndarray) -> np.ndarray:
        corr = self.mu_to_corr(theta[0])
        theta = np.vstack([self.corr_to_general_product(corr)] * self.l)
        return theta

    def connectedness_to_geometric(self, theta: np.ndarray) -> np.ndarray:
        q = np.mean([self.log_mu_to_q(theta_i) for theta_i in theta])
        theta = np.array([[self.q_to_log_mu(q)]])
        return theta

    def connectedness_to_jenga(self, theta: np.ndarray) -> np.ndarray:
        z = np.zeros((self.l, self.n_alleles))
        return np.hstack([theta, z])

    def connectedness_to_general_product(
        self, theta: np.ndarray
    ) -> np.ndarray:
        theta = np.vstack(
            [
                self.corr_to_general_product(self.mu_to_corr(theta_i))
                for theta_i in theta
            ]
        )
        return theta

    def jenga_to_corr(self, theta: np.ndarray) -> np.ndarray:
        log_mu, log_p = theta[0], theta[1:]
        mu = np.exp(log_mu)
        p = np.exp(log_p - logsumexp(log_p))
        eta = (1 - p) / p
        fs = np.sqrt(1 + eta * mu)
        corr = (1 - mu.reshape((1, 1))) / (
            np.expand_dims(fs, 0) * np.expand_dims(fs, 1)
        )
        np.fill_diagonal(corr, 1)
        return corr

    def jenga_to_connectedness(self, theta: np.ndarray) -> np.ndarray:
        qs = [self.corr_to_q(self.jenga_to_corr(theta_i)) for theta_i in theta]
        theta = np.array([[self.q_to_log_mu(q)] for q in qs])
        return theta

    def jenga_to_geometric(self, theta: np.ndarray) -> np.ndarray:
        q = np.mean(
            [self.corr_to_q(self.jenga_to_corr(theta_i)) for theta_i in theta]
        )
        theta = np.array([[self.q_to_log_mu(q)]])
        return theta

    def jenga_to_general_product(self, theta: np.ndarray) -> np.ndarray:
        theta = np.vstack(
            [
                self.corr_to_general_product(self.jenga_to_corr(theta_i))
                for theta_i in theta
            ]
        )
        return theta

    def general_product_to_geometric(self, theta: np.ndarray) -> np.ndarray:
        q = np.mean(
            [
                self.corr_to_q(self.general_product_to_corr(theta_i))
                for theta_i in theta
            ]
        )
        theta = np.array([[self.q_to_log_mu(q)]])
        return theta

    def general_product_to_connectedness(
        self, theta: np.ndarray
    ) -> np.ndarray:
        qs = [
            self.corr_to_q(self.general_product_to_corr(theta_i))
            for theta_i in theta
        ]
        theta = np.array([[self.q_to_log_mu(q)] for q in qs])
        return theta

    def general_product_to_jenga(self, theta: np.ndarray) -> np.ndarray:
        thetas: List[np.ndarray] = []
        for theta_i in theta:
            B = self.general_product_to_corr(theta_i)

            def loss(params: np.ndarray) -> float:
                A = self.jenga_to_corr(params)
                return self.calc_frob(A, B)

            params0 = np.zeros(self.n_alleles + 1)
            res = minimize(loss, x0=params0)
            thetas.append(res.x)
        return np.vstack(thetas)


class LinearEmbeddingKernel(CorrelationKernel):
    def __init__(
        self,
        seq_length: Optional[int] = None,
        alphabet_type: Optional[str] = None,
        alphabet: Optional[List[str]] = None,
        alphabet_list: Optional[List[List[str]]] = None,
        use_keops: bool = False,
        train_embedding: bool = True,
    ):
        super().__init__(
            seq_length=seq_length,
            alphabet_type=alphabet_type,
            alphabet=alphabet,
            alphabet_list=alphabet_list,
            use_keops=use_keops,
        )
        self.n_features_free = self.n_features - self.l
        self.M_shape = self.n_features, self.n_features
        self.P0 = self.calc_P()
        self.train_embedding = train_embedding

    def calc_P(self):
        Ps = [torch.eye(a) - torch.ones((a, a)) / a for a in self.alphas]
        P = torch.block_diag(*Ps)
        U, S, _ = torch.linalg.svd(P, full_matrices=False)
        P = U[:, S > 1e-4]
        return P

    def get_M(self):
        A = self.get_A()
        M = A @ A.T
        if hasattr(self, "get_diag"):
            i = torch.arange(self.n_features)
            M[i, i] = torch.diag(M) + self.get_diag()
        return M

    def _nonkeops_forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        A = self.get_A()
        z1 = self.x_to_z(x1, A)
        z2 = self.x_to_z(x2, A)
        diff1 = diff_inner_prod(z1, z2, diag=diag)
        diff2 = diff_inner_prod(z2, z1, diag=diag).T
        diff = 0.5 * (diff1 + diff2)
        kernel = torch.exp(self.log_var - diff)
        return kernel

    def _covar_func(
        self, x1: torch.Tensor, x2: torch.Tensor, **kwargs: Any
    ) -> LazyTensor:
        x1_ = LazyTensor(x1[:, None, :])
        x2_ = LazyTensor(x2[None, :, :])
        K = (-((x1_ - x2_) ** 2).sum(-1)).exp()
        return K

    def x_to_z(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        return x @ A

    def _keops_forward(
        self, x1: torch.Tensor, x2: torch.Tensor, **kwargs: Any
    ) -> KernelLinearOperator:
        sigma2 = torch.exp(self.log_var)
        A = self.get_A()
        z1 = self.x_to_z(x1, A)
        z2 = self.x_to_z(x2, A)
        kernel = sigma2 * KernelLinearOperator(
            z1, z2, covar_func=self._covar_func, **kwargs
        )
        return kernel

    def get_M_decay_factors(self):
        M = self.get_M().detach().numpy()

        # Iterate for each pair of positions
        decay_factors = {}
        for p, q in product(self.positions, repeat=2):
            alpha_p = self.alphas[p]
            alpha_q = self.alphas[q]
            alleles_p = list(range(alpha_p))
            alleles_q = list(range(alpha_q))
            allele_pairs_p = list(product(alleles_p, repeat=2))
            allele_pairs_q = list(product(alleles_q, repeat=2))

            # Fill in matrix for a pair of positions
            decay_factors_pq = np.zeros((alpha_p, alpha_q))

            # diagonal blocks
            if p == q:
                for i, (x, y) in enumerate(allele_pairs_p):
                    s, e = self.starts[p], self.ends[p]
                    M_pp = M[s:e, :][:, s:e]
                    delta = 1 - np.exp(
                        2 * M_pp[x, y] - M_pp[x, x] - M_pp[y, y]
                    )
                    decay_factors_pq[i, i] = delta

            # off-diagonal blocks
            else:
                for i, (x_p, y_p) in enumerate(allele_pairs_p):
                    for j, (x_q, y_q) in enumerate(allele_pairs_q):
                        s1, e1 = self.starts[p], self.ends[p]
                        s2, e2 = self.starts[q], self.ends[q]
                        M_pq = M[s1:e1, :][:, s2:e2]
                        delta = 1 - np.exp(
                            M_pq[x_p, y_q]
                            + M_pq[y_p, x_q]
                            - M_pq[x_p, x_q]
                            - M_pq[y_p, y_q]
                        )
                        decay_factors_pq[i, j] = delta

            decay_factors[(p, q)] = pd.DataFrame(
                decay_factors_pq,
                index=self.alphabet_list[p],
                columns=self.alphabet_list[q],
            )
        return decay_factors


class MahalanobisRBFKernel(LinearEmbeddingKernel):
    def __init__(
        self,
        seq_length: Optional[int] = None,
        alphabet_type: Optional[str] = None,
        alphabet: Optional[List[str]] = None,
        alphabet_list: Optional[List[List[str]]] = None,
        use_keops: bool = False,
        log_var0: Optional[torch.Tensor] = None,
        log_diag_sqrt0: Optional[torch.Tensor] = None,
        theta0: Optional[torch.Tensor] = None,
        M0: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            seq_length=seq_length,
            alphabet_type=alphabet_type,
            alphabet=alphabet,
            alphabet_list=alphabet_list,
            use_keops=use_keops,
        )
        self.theta_dim = int(comb(self.n_features_free, 2))
        self.theta_to_L = CorrCholeskyTransform()
        self._set_params(log_var0, log_diag_sqrt0, theta0, M0)

    def get_theta0_log_sqrt_diag(
        self,
        theta0: Optional[torch.Tensor] = None,
        M0: Optional[torch.Tensor] = None,
        log_diag_sqrt0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        log_diag_sqrt0 = None
        if theta0 is not None and M0 is not None:
            msg = "Either "
            raise ValueError(msg)
        elif theta0 is None:
            if M0 is None:
                v = np.ones(self.n_features)
                M0 = torch.diag(torch.Tensor(v))

            if M0.shape != self.M_shape:
                msg = f"M0 should have shape {self.M_shape} but got {M0.shape} instead"
                raise ValueError(msg)

            B = torch.linalg.inv(self.P0.T @ self.P0) @ self.P0.T
            M_prime = B @ M0 @ B.T
            log_diag_sqrt0 = torch.log(
                torch.sqrt(torch.diag(M_prime))
            ).unsqueeze(1)
            L = torch.linalg.cholesky(M_prime)
            theta0 = self.theta_to_L._inverse(L)

        if theta0.shape != (self.theta_dim,):
            msg = f"theta0 should have shape {(self.theta_dim,)} but got {theta0.shape}"
            raise ValueError(msg)

        if log_diag_sqrt0 is None:
            log_diag_sqrt0 = torch.zeros(self.n_features_free, 1)

        if log_diag_sqrt0.shape != (self.n_features_free, 1):
            msg = f"theta0 should have shape {(self.n_features_free, 1)} but got {log_diag_sqrt0.shape}"
            raise ValueError(msg)

        return (theta0, log_diag_sqrt0)

    def _set_params(
        self,
        log_var0: Optional[torch.Tensor] = None,
        log_diag_sqrt0: Optional[torch.Tensor] = None,
        theta0: Optional[torch.Tensor] = None,
        M0: Optional[torch.Tensor] = None,
    ):
        log_var0 = self.get_log_var0(log_var0)
        theta0, log_diag_sqrt0 = self.get_theta0_log_sqrt_diag(
            theta0, M0, log_diag_sqrt0
        )

        P = Parameter(self.P0, requires_grad=False)
        theta = Parameter(theta0, requires_grad=self.train_embedding)
        log_diag_sqrt = Parameter(
            log_diag_sqrt0, requires_grad=self.train_embedding
        )
        log_var = Parameter(log_var0, requires_grad=True)

        self.register_parameter(name="P", parameter=P)
        self.register_parameter(name="log_diag_sqrt", parameter=log_diag_sqrt)
        self.register_parameter(name="theta", parameter=theta)
        self.register_parameter(name="log_var", parameter=log_var)

    def get_A(self):
        diag_sqrt = torch.exp(self.log_diag_sqrt)
        L = diag_sqrt * self.theta_to_L(self.theta)
        A = self.P @ L
        return A


class FactorAnalysisKernel(LinearEmbeddingKernel):
    def __init__(
        self,
        seq_length: Optional[int] = None,
        alphabet_type: Optional[str] = None,
        alphabet: Optional[List[str]] = None,
        alphabet_list: Optional[List[List[str]]] = None,
        use_keops: bool = False,
        train_embedding: bool = True,
        log_var0: Optional[torch.Tensor] = None,
        A0: Optional[torch.Tensor] = None,
        ndim: int = 3,
    ):
        super().__init__(
            seq_length=seq_length,
            alphabet_type=alphabet_type,
            alphabet=alphabet,
            alphabet_list=alphabet_list,
            train_embedding=train_embedding,
            use_keops=use_keops,
        )
        self.ndim = ndim
        self._set_params(log_var0=log_var0, A0=A0)

    def get_A(self) -> torch.Tensor:
        q = torch.linalg.qr(self.q)[0]
        lda = torch.exp(self.log_sqrt_lda)
        A = self.P @ (q * lda)
        return A

    def get_q0_log_sqrt_lda0(
        self, A0: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = (self.n_features, self.ndim)
        if A0 is None:
            A0 = torch.randn(shape)

        if A0.shape != shape:
            msg = f"A0 should have shape {shape} but got {A0.shape}"
            raise ValueError(msg)

        M0 = A0 @ A0.T
        B = torch.linalg.inv(self.P0.T @ self.P0) @ self.P0.T
        M_prime = B @ M0 @ B.T
        lda, q = torch.linalg.eigh(M_prime)
        log_sqrt_lda0 = 0.5 * torch.log(lda[-self.ndim :]).unsqueeze(0)
        q0 = q[:, -self.ndim :]
        return (q0, log_sqrt_lda0)

    def _set_params(
        self,
        log_var0: Optional[torch.Tensor] = None,
        A0: Optional[torch.Tensor] = None,
    ):
        log_var0 = self.get_log_var0(log_var0)
        q0, log_sqrt_lda0 = self.get_q0_log_sqrt_lda0(A0)

        P = Parameter(self.P0, requires_grad=False)
        q = Parameter(q0, requires_grad=self.train_embedding)
        log_sqrt_lda = Parameter(
            log_sqrt_lda0, requires_grad=self.train_embedding
        )
        log_var = Parameter(log_var0, requires_grad=True)

        self.register_parameter(name="P", parameter=P)
        self.register_parameter(name="q", parameter=q)
        self.register_parameter(name="log_sqrt_lda", parameter=log_sqrt_lda)
        self.register_parameter(name="log_var", parameter=log_var)


class ConnectednessFactorAnalysisKernel(LinearEmbeddingKernel):
    def __init__(
        self,
        seq_length: Optional[int] = None,
        alphabet_type: Optional[str] = None,
        alphabet: Optional[List[str]] = None,
        alphabet_list: Optional[List[List[str]]] = None,
        use_keops: bool = False,
        log_var0: Optional[torch.Tensor] = None,
        C0: Optional[torch.Tensor] = None,
        train_embedding: bool = True,
    ):
        super().__init__(
            seq_length=seq_length,
            alphabet_type=alphabet_type,
            alphabet=alphabet,
            alphabet_list=alphabet_list,
            use_keops=use_keops,
            train_embedding=train_embedding,
        )
        self.theta_to_L = LowerCholeskyTransform()
        self._set_params(log_var0, C0)

    def get_theta0(self, C0: Optional[torch.Tensor]) -> torch.Tensor:
        shape = self.l, self.l
        if C0 is None:
            C0 = torch.eye(self.l)

        if C0.shape != shape:
            msg = f"C0 should have shape {shape} but got {C0.shape}"
            raise ValueError(msg)

        L0 = cholesky(C0)
        theta0 = self.theta_to_L._inverse(L0)
        return theta0

    @property
    def A(self):
        return self.theta_to_L(self.theta)

    def _set_params(
        self,
        log_var0: Optional[torch.Tensor] = None,
        C0: Optional[torch.Tensor] = None,
    ):
        log_var0 = self.get_log_var0(log_var0)
        theta0 = self.get_theta0(C0)

        theta = Parameter(theta0, requires_grad=True)
        log_var = Parameter(log_var0, requires_grad=True)

        self.register_parameter(name="theta", parameter=theta)
        self.register_parameter(name="log_var", parameter=log_var)

    def x_to_z(self, x: torch.Tensor) -> torch.Tensor:
        A = self.A
        z = torch.zeros_like(x)
        for allele in self.alphabet_full:
            indices = self.alleles_idx[allele]
            sites = self.alleles_pos[allele]
            z[:, indices] = x[:, indices] @ A[sites, :][:, sites]
        return z

    def get_C(self):
        return self.A @ self.A.T  # type: ignore


def get_named_kernel(
    kernel: str, alphabet_list: List[List[str]], kwargs: dict = {}
) -> Kernel:
    NAMED_KERNELS = {
        "Additive": AdditiveKernel,
        "Pairwise": PairwiseKernel,
        "VC": VarianceComponentKernel,
        "VarianceComponent": VarianceComponentKernel,
        "Geometric": GeometricKernel,
        "Connectedness": ConnectednessKernel,
        "Jenga": JengaKernel,
        "GeneralProduct": GeneralProductKernel,
        "FactorAnalysis": FactorAnalysisKernel,
        "MahalanobisRBF": MahalanobisRBFKernel,
    }
    if kernel in NAMED_KERNELS:
        return NAMED_KERNELS[kernel](alphabet_list=alphabet_list, **kwargs)
    else:
        msg = (
            f"kernel label {kernel} not allowed. Check {NAMED_KERNELS.keys()}"
        )
        raise ValueError(msg)
