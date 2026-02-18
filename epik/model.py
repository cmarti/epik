import sys
from copy import deepcopy
from time import time
from typing import Any, Optional, Tuple, List, Union

import numpy as np
import pandas as pd
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, _GaussianLikelihoodBase
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.mlls import MarginalLogLikelihood, VariationalELBO
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.settings import (
    cg_tolerance,
    eval_cg_tolerance,
    fast_pred_var,
    fast_computations,
    max_cg_iterations,
    max_lanczos_quadrature_iterations,
    max_preconditioner_size,
    max_cholesky_size,
    max_root_decomposition_size,
    num_likelihood_samples,
    num_trace_samples,
    skip_posterior_variances,
)
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    UnwhitenedVariationalStrategy,
)
from torch.optim import Adam
from tqdm import tqdm

from epik.kernel import get_named_kernel, SiteProductKernel
from epik.utils import (
    get_epistatic_coeffs_contrast_matrix,
    get_mut_effs_contrast_matrix,
    get_tensor,
    to_numpy,
    validate_alphabet,
    get_one_hot_encoding,
    split_training_test,
)


class ExactMLL(MarginalLogLikelihood):
    """
    Adapted from GPyTorch's implementation to report the complete
    rather than by point average marginal log-likelihood for
    better interpretation of differences in their values.


    Parameters
    ----------
    likelihood : likelihood function
        Likelihood function from the family of Gaussian likelihoods
        either with fixed or trainable error models p(y|f).

    model : GPmodel
        Gaussian process model over the function value p(f).

    """

    def __init__(self, likelihood, model) -> None:
        if not isinstance(likelihood, _GaussianLikelihoodBase):
            raise RuntimeError("Likelihood must be Gaussian for exact inference")
        super(ExactMLL, self).__init__(likelihood, model)

    def _add_other_terms(self, res: torch.Tensor, params: tuple) -> torch.Tensor:
        # Add additional terms (SGPR / learned inducing points, heteroskedastic likelihood models)
        for added_loss_term in self.model.added_loss_terms():
            res = res.add(added_loss_term.loss(*params))

        # Add log probs of priors on the (functions of) parameters
        res_ndim = res.ndim
        for name, module, prior, closure, _ in self.model.named_priors():
            prior_term = prior.log_prob(closure(module))
            res.add_(prior_term.view(*prior_term.shape[:res_ndim], -1).sum(dim=-1))

        return res

    def forward(
        self, function_dist: MultivariateNormal, y: torch.Tensor, *params: Any
    ) -> torch.Tensor:
        """
        Computes the exact marginal log likelihood.

        Parameters
        ----------
        function_dist : MultivariateNormal
            The Gaussian random variable representing the function distribution.
        y : torch.Tensor
            The y tensor for which the log probability is computed.
        *params : Any
            Additional parameters required by the likelihood.

        Returns
        -------
        torch.Tensor
            The computed log probability of the y under the marginal
            distribution, with additional terms added if applicable.

        Raises
        ------
        RuntimeError
            If `function_dist` is not an instance of `MultivariateNormal`.
        """

        if not isinstance(function_dist, MultivariateNormal):
            raise RuntimeError(
                "ExactMarginalLogLikelihood can only operate on Gaussian random variables"
            )

        # Get the log prob of the marginal distribution
        output = self.likelihood(function_dist, *params)
        res = output.log_prob(y)
        res = self._add_other_terms(res, params)
        return res


class GPModel(ExactGP):
    """
    Gaussian process model class.

    This class returns the Multivariate Gaussian distribution representing
    the posterior probability of a function defined over a set of input points
    x conditioned on some observations under a given kernel function. The model
    can use either a constant mean or zero mean function.

    train_x : torch.Tensor
        The input training data (features).
    train_y : torch.Tensor
        The observed training data (targets).
    kernel : gpytorch.kernels.Kernel
        The kernel function defining the covariance structure of the GP.
    likelihood : gpytorch.likelihoods.Likelihood
        The likelihood function for the GP model.
    constant_mean : float, optional, default=0.0
        Constant mean function to use.
    train_mean : bool, optional, default=False
        If True, includes a trainable mean function in addition to the
        constant_mean.
    """

    def __init__(
        self,
        kernel: Kernel,
        train_x: Optional[torch.Tensor] = None,
        train_y: Optional[torch.Tensor] = None,
        likelihood: Optional[_GaussianLikelihoodBase] = None,
        constant_mean: float = 0.0,
        train_mean: bool = False,
    ) -> None:
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean() if train_mean else ZeroMean()
        self.covar_module = kernel
        self.constant_mean = constant_mean

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """
        Computes the posterior distribution for the input `x`.

        x : torch.Tensor
            The input data for which the posterior distribution is computed.

        Returns
        -------
        gpytorch.distributions.MultivariateNormal
            The posterior distribution as a multivariate normal distribution.
        """
        mean_x = self.constant_mean + self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class GeneralizedGPModel(ApproximateGP):
    """
    Generalized Gaussian process model class.

    This class represents a generalized Gaussian process model that uses
    variational inference for approximate posterior estimation. It returns
    the Multivariate Gaussian distribution representing the posterior
    probability of a function defined over a set of input points `x`
    conditioned on some observations under a given kernel function.

    Parameters
    ----------
    train_x : torch.Tensor
        The input training data (features).
    kernel : gpytorch.kernels.Kernel
        The kernel function defining the covariance structure of the GP.
    train_mean : bool, optional, default=False
        If True, use a constant mean function; otherwise, use a zero mean function.
    """

    def __init__(
        self, train_x: torch.Tensor, kernel: Kernel, train_mean: bool = False
    ) -> None:
        distribution = CholeskyVariationalDistribution(train_x.size(0))
        strategy = UnwhitenedVariationalStrategy(
            self, train_x, distribution, learn_inducing_locations=False
        )
        super(GeneralizedGPModel, self).__init__(strategy)
        self.mean_module = ConstantMean() if train_mean else ZeroMean()
        self.covar_module: Any = kernel

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """
        Computes the approximate posterior distribution for the input `x`.

        x : torch.Tensor
            The input data for which the posterior distribution is computed.

        Returns
        -------
        gpytorch.distributions.MultivariateNormal
            The posterior distribution as a multivariate normal distribution.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class _Epik(object):
    def __init__(
        self,
        kernel: Union[str, Kernel],
        seq_length: Optional[int] = None,
        alphabet_type: Optional[str] = None,
        alphabet: Optional[List[str]] = None,
        alphabet_list: Optional[List[List[str]]] = None,
        device: str = "cpu",
        train_mean: bool = False,
        train_noise: bool = False,
        constant_mean: float = 0.0,
        kernel_kwargs: dict = {},
    ) -> None:
        self.alphabet_list = validate_alphabet(
            seq_length, alphabet_type, alphabet, alphabet_list
        )
        self.extended_alphabet_list = [
            ["*"] + alphabet for alphabet in self.alphabet_list
        ]
        self.l = len(self.alphabet_list)
        self.kernel = self.get_kernel(kernel, kernel_kwargs)
        self.device = device
        self.train_mean = train_mean
        self.train_noise = train_noise
        self.constant_mean = constant_mean

    def get_kernel(
        self,
        kernel: Union[str, Kernel],
        kwargs: dict,
    ) -> Kernel:
        """
        Retrieve or initialize a kernel for the Gaussian process model.

        Parameters
        ----------
        kernel : str or Kernel
            The kernel to use. It can be either a string representing the name
            of a predefined kernel or an instance of a Kernel object.
        kwargs : dict
            Additional arguments to pass to the kernel initialization, such
            as kernel hyperparameters.

        Returns
        -------
        Kernel
            The initialized or retrieved kernel object.

        Raises
        ------
        ValueError
            If the kernel name is not recognized or the provided kernel object
            is invalid.
        """

        if isinstance(kernel, str):
            kernel = get_named_kernel(kernel, self.alphabet_list, kwargs=kwargs)
        elif not isinstance(kernel, Kernel):
            msg = "`kernel` must be either a kernel name or a pre-defined Kernel object"
            raise ValueError(msg)

        if hasattr(kernel, "alphabet_list"):
            if self.alphabet_list != kernel.alphabet_list:
                msg = f"The alphabet in the provided kernel {kernel.alphabet_list} "
                msg += f"does not match the model alphabet {self.alphabet_list}"

        return kernel

    def encode(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        return self.get_tensor(get_one_hot_encoding(X, self.alphabet_list))

    def report_fit_progress(self, pbar: Any) -> None:
        allocated: float = torch.cuda.memory_allocated(device=self.device) / 1e6
        reserved: float = torch.cuda.memory_reserved(device=self.device) / 1e6
        report_dict: dict[str, str] = {
            "MLL": f"{self.mll:.3f}",
            "Mem(alloc/res)": f"{allocated:.2f}/{reserved:.2f}MB",
        }
        if hasattr(self, "scheduler"):
            report_dict["LR"] = f"{self.optimizer.param_groups[0]['lr']:.4f}"

        pbar.set_postfix(report_dict)

    def get_tensor(self, ndarray: Any) -> torch.Tensor:
        return get_tensor(ndarray, device=self.device)

    def set_training_mode(self) -> None:
        self.gp.train()
        self.likelihood.train()

    def set_evaluation_mode(self) -> None:
        self.gp.eval()
        self.likelihood.eval()

    def set_data(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        y_var: Union[np.ndarray, torch.Tensor, None] = None,
    ) -> None:
        """
        Set the training data for the model.

        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            A array or tensor of shape (n_sequences,) containing the
            input sequences as str.

        y : torch.Tensor
            A tensor of shape (n_sequences,) containing the phenotypic
            measurements corresponding to each sequence in `X`.

        y_var : torch.Tensor, optional
            A tensor of shape (n_sequences,) representing the variance
            of the measurements in `y`. If `None`, it is assumed that
            there is no uncertainty in the measurements.
        """
        self.X_seqs = X
        self.X = self.get_tensor(get_one_hot_encoding(X, self.alphabet_list))
        self.y = self.get_tensor(y)
        if y_var is None:
            self.y_var: torch.Tensor = torch.zeros_like(self.y)
        else:
            self.y_var = self.get_tensor(y_var)

        if self.X.shape[0] != self.y.shape[0] or self.y.shape[0] != self.y_var.shape[0]:
            msg = "X, y and y_var should have the same size {}, {}, {}"
            raise ValueError(
                msg.format(self.X.shape[0], self.y.shape[0], self.y_var.shape[0])
            )

        if self.X.shape[1] != self.kernel.n_features:
            msg = "Number of features in X ({}) should match the kernel features ({})"
            raise ValueError(msg.format(self.X.shape[1], self.kernel.n_features))

        self.define_likelihood()
        self.define_gp()
        self.n = X.shape[0]

    def get_fast_comp(self, method) -> bool:
        if method == "cg":
            return True
        elif method == "cholesky":
            return False
        else:
            msg = "Invalid method provided. Must be either cholesky or cg"
            raise ValueError(msg)

    def calc_mll(
        self,
        method: str = "cg",
        cg_tol: float = 0.1,
        max_cg_iter: int = 1000,
        n_lanczos_iter: int = 50,
        n_trace_samples: int = 50,
        preconditioner_size: int = 0,
    ) -> torch.Tensor:
        """
        Calculate the marginal log likelihood (MLL) for the Gaussian process model.

        This method computes the MLL using either the conjugate gradient (CG) method
        or the Cholesky decomposition, depending on the specified `method`. It also
        allows for customization of various parameters such as tolerance, maximum
        iterations, preconditioner size, and the number of trace samples.

        Parameters
        ----------
        method : str, optional
            The method to use for MLL computation. Must be one of ["cg", "cholesky"].
        cg_tol : float, optional
            The tolerance for the conjugate gradient method.
        max_cg_iter : int, optional
            The maximum number of conjugate gradient iterations.
        n_lanczos_iter : int, optional
            The number of Lanczos iterations for stochastic trace estimation.
        n_trace_samples : int, optional
            The number of trace samples to use for stochastic trace estimation.
        preconditioner_size : int, optional
            The size of the preconditioner used to accelerate convergence.

        Returns
        -------
        torch.Tensor
            The computed marginal log likelihood as a tensor.

        Notes
        -----
        This method uses context managers to temporarily set various parameters
        during the computation of the MLL. The `max_preconditioner_size`,
        `cg_tolerance`, `num_trace_samples`, `max_lanczos_quadrature_iterations`,
        and `max_cholesky_size` context managers are used to manage these settings.
        """
        allowed: list[str] = ["cg", "cholesky"]
        if method not in allowed:
            raise ValueError(f"method {method} should be one of {allowed}")
        fast = self.get_fast_comp(method)

        with max_preconditioner_size(preconditioner_size), cg_tolerance(
            cg_tol
        ), num_trace_samples(n_trace_samples), max_lanczos_quadrature_iterations(
            n_lanczos_iter
        ), fast_computations(log_prob=fast), max_cg_iterations(
            max_cg_iter
        ), max_cholesky_size(1):
            return self.mll_layer(self.gp(self.X), self.y)

    def diagnose_mll(
        self,
        cg_tol: float = 1.0,
        max_cg_iter: int = 1000,
        n_lanczos_iter: int = 50,
        preconditioner_size: int = 0,
        n_trace_samples: int = 50,
        min_n_lanczos: int = 20,
        max_n_lanczos: int = 1000,
        min_cg_tol: float = 0.001,
        max_cg_tol: float = 10,
        min_n_trace_samples: int = 10,
        max_n_trace_samples: int = 200,
        add_cholesky: bool = False,
        track_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Diagnose the marginal log likelihood (MLL) by varying the number of
        Lanczos iterations, CG tolerance, and the number of trace samples,
        and recording the resulting MLL values.

        Parameters
        ----------
        min_n_lanczos : int
            Minimum number of Lanczos iterations to test.
        max_n_lanczos : int
            Maximum number of Lanczos iterations to test.
        min_cg_tol : float
            Minimum CG tolerance to test.
        max_cg_tol : float
            Maximum CG tolerance to test.
        min_n_trace_samples : int
            Minimum number of trace samples to test.
        max_n_trace_samples : int
            Maximum number of trace samples to test.
        add_cholesky : bool
            Whether to compute MLL with Cholesky decomposition as well.
        track_progress : bool
            Whether to track progress.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the number of Lanczos iterations
            (`n_lanczos`), CG tolerance (`cg_tol`), and
            number of samples for stochastic trace estimation (`n_trace_samples`)
            with the corresponding MLL values (`mll`).
        """
        max_value = max(n_lanczos_iter + 50, max_n_lanczos)
        records = []
        ns = np.linspace(min_n_lanczos, max_value, 50).astype(int)
        cg_tols = np.geomspace(min_cg_tol, max_cg_tol, 50)
        ns_trace_samples = np.linspace(
            min_n_trace_samples, max_n_trace_samples, 50
        ).astype(int)

        if track_progress:
            ns = tqdm(ns)
            cg_tols = tqdm(cg_tols)
            ns_trace_samples = tqdm(ns_trace_samples)

        kwargs = {
            "method": "cg",
            "cg_tol": cg_tol,
            "max_cg_iter": max_cg_iter,
            "n_lanczos_iter": n_lanczos_iter,
            "n_trace_samples": n_trace_samples,
            "preconditioner_size": preconditioner_size,
        }
        params = {
            "n_lanczos_iter": ns,
            "cg_tol": cg_tols,
            "n_trace_samples": ns_trace_samples,
        }
        with torch.inference_mode():
            for param, values in params.items():
                for value in values:
                    record = kwargs.copy()
                    record[param] = value
                    record["mll"] = self.calc_mll(**record).item()
                    record["param"] = param
                    records.append(record)

            if add_cholesky:
                record = kwargs.copy()
                record["mll"] = self.calc_mll(**record).item()
                record["method"] = "cholesky"
                records.append(record)

        return pd.DataFrame(records)

    def training_step(
        self,
        mll_method: str = "cg",
        cg_tol: float = 1.0,
        max_cg_iter: int = 1000,
        n_lanczos_iter: int = 50,
        n_trace_samples: int = 50,
        preconditioner_size: int = 0,
    ) -> None:
        torch.cuda.empty_cache()
        mll = self.calc_mll(
            method=mll_method,
            cg_tol=cg_tol,
            max_cg_iter=max_cg_iter,
            n_lanczos_iter=n_lanczos_iter,
            n_trace_samples=n_trace_samples,
            preconditioner_size=preconditioner_size,
        )
        self.mll: float = mll.detach().item()

        skip_grad = False
        if len(self.training_history) > 20:
            sd = np.std(self.training_history[-10:])
            threshold = self.training_history[-1] - 10 * sd

            # Only update gradient if MLL is safe
            if self.mll < threshold:
                print(self.mll, threshold, self.training_history[0])
                msg: str = f"Gradient calculation skipped due to unusually low MLL={mll.item()}"
                sys.stderr.write(msg)
                skip_grad = True

        if not skip_grad:
            self.optimizer.zero_grad()
            mll.backward()

        self.optimizer.step()
        self.params = self.gp.state_dict()

        params = {}
        grad = {}
        for name, param in self.mll_layer.named_parameters():
            new_name = name.split(".")[-1]
            if param.grad is not None:
                params[new_name] = param.detach().cpu().numpy()
                grad[new_name] = param.grad.detach().cpu().numpy()

        self.params_history.append(params)
        self.grad_history.append(grad)
        self.training_history.append(self.mll)

        if not hasattr(self, "max_mll") or self.mll > self.max_mll:
            self.max_mll = self.mll
            self.max_params = deepcopy(self.params)

    def fit(
        self,
        n_iter: int = 100,
        learning_rate: float = 0.1,
        mll_method: str = "cg",
        cg_tol: float = 1.0,
        max_cg_iter: int = 1000,
        n_lanczos_iter: int = 50,
        n_trace_samples: int = 50,
        preconditioner_size: int = 0,
        track_progress: bool = False,
    ) -> None:
        """
        Optimize model hyperparameters by maximizing the marginal log-likelihood.

        Parameters
        ----------
        n_iter : int, optional (default=100)
            Number of iterations for the optimization process.
        learning_rate : float, optional (default=0.1)
            Learning rate for the optimizer.
        mll_method : str, optional (default="cg")
            The method to use for MLL computation. Must be one of ["cg", "cholesky"].
        cg_tol : float, optional (default=1.0)
            The tolerance for the conjugate gradient method.
        max_cg_iter : int, optional (default=1000)
            The maximum number of conjugate gradient iterations.
        n_lanczos_iter : int, optional (default=50)
            The maximum number of Lanczos iterations.
        preconditioner_size : int, optional (default=0)
            The size of the preconditioner.
        n_trace_samples : int, optional (default=50)
            The number of trace samples to use.

        Raises
        ------
        RuntimeError
            If an out-of-memory error occurs during training and cannot be resolved.
        """
        self.training_history: list[float] = []
        self.params_history: list[dict[str, Any]] = []
        self.grad_history: list[dict[str, Any]] = []
        self.set_training_mode()
        self.optimizer = Adam(self.gp.parameters(), lr=learning_rate, maximize=True)

        t0: float = time()
        pbar = range(n_iter)
        if n_iter > 1 and track_progress:
            pbar: tqdm = tqdm(pbar, desc="Optimizing hyperparameters")

        for _ in pbar:
            try:
                self.training_step(
                    mll_method=mll_method,
                    cg_tol=cg_tol,
                    max_cg_iter=max_cg_iter,
                    n_lanczos_iter=n_lanczos_iter,
                    n_trace_samples=n_trace_samples,
                    preconditioner_size=preconditioner_size,
                )
            except RuntimeError as error:
                if "out of memory" in str(error):
                    torch.cuda.empty_cache()
                    self.kernel.use_keops = True
                    self.training_step(
                        mll_method=mll_method,
                        cg_tol=cg_tol,
                        max_cg_iter=max_cg_iter,
                        n_lanczos_iter=n_lanczos_iter,
                        n_trace_samples=n_trace_samples,
                        preconditioner_size=preconditioner_size,
                    )
                else:
                    raise RuntimeError(error)

            if n_iter > 1 and track_progress:
                self.report_fit_progress(pbar)

        self.fit_time: float = time() - t0

    @property
    def history(self) -> pd.DataFrame:
        return pd.DataFrame({"mll": self.training_history})

    def get_params(self) -> dict:
        return self.gp.state_dict()

    def save(self, fpath: str) -> None:
        """
        Save the model parameters to a file for future use.

        Parameters
        ----------
        fpath : str
            The file path where the model parameters will be saved.
        """
        torch.save(self.gp.state_dict(), fpath)

    def set_params(self, params: dict) -> None:
        self.gp.load_state_dict(params)

    def load(self, fpath: str, **kwargs: Any) -> None:
        """
        Load model parameters from a file.

        Parameters
        ----------
        fpath : str
            Path to the file containing the stored model parameters.

        **kwargs : dict, optional
            Additional arguments to pass to `torch.load` for loading the parameters.
        """
        params = torch.load(fpath, **kwargs)
        self.set_params(params)


class EpiK(_Epik):
    """
    Gaussian process regression model for inferring
    sequence-function relationships from experimental measurements
    using GPyTorch and KeOps backend.

    Parameters
    ----------
    kernel : epik.kernel.Kernel
        An instance of a kernel class that defines the covariance
        structure between pairs of sequences for Gaussian process regression.

    device : str, optional
        The device on which computations will be performed. Options are
        "cpu" or "cuda". Default is "cpu".

    constant_mean : float, optional
        Value of the prior mean to use for the Gaussian process model.
        Default is 0. If `train_mean=True`, then this value initializes
        the mean function to learn.

    train_mean : bool, optional
        Whether to optimize the mean function of the Gaussian Process.
        By default, it assumes a zero-mean function. Default is False.

    train_noise : bool, optional
        Whether to learn an additional noise parameter for the Gaussian Process.
        By default, it assumes that the provided error estimates are reliable.
        Default is False.

    preconditioner_size : int, optional
        The size of the preconditioner used to accelerate
        conjugate gradient convergence. By default, no
        preconditioner is computed. Default is 0.

    cg_tol : float, optional
        The tolerance level for the conjugate gradient solver. This parameter
        controls the precision of the solver. Lower values result in higher
        precision but may increase computation time. Default is 1.0.

    num_trace_samples : int, optional
        The number of samples used to estimate the trace of a matrix during
        computations. Increasing this value improves the accuracy of the trace
        estimation but increases computational cost. Default is 50.

    max_n_lanczos_iterations : int, optional
        The maximum number of Lanczos iterations to perform during matrix
        decompositions. Higher values may improve accuracy but increase
        computation time. Default is 50.

    track_progress : bool, optional
        Whether to display a progress bar during model fitting. Default is False.

    """

    def define_likelihood(self) -> None:
        self.likelihood = FixedNoiseGaussianLikelihood(
            noise=self.y_var, learn_additional_noise=self.train_noise
        )
        if self.device != "cpu":
            self.likelihood = self.likelihood.to(device=self.device)

    def get_gp(
        self,
        likelihood: _GaussianLikelihoodBase,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> GPModel:
        gp = GPModel(
            self.kernel,
            x,
            y,
            likelihood,
            train_mean=self.train_mean,
            constant_mean=self.constant_mean,
        )

        if self.device == "cuda":
            gp = gp.cuda()

        return gp

    def define_gp(self) -> None:
        self.gp: GPModel = self.get_gp(self.likelihood, self.X, self.y)
        self.mll_layer: ExactMLL = ExactMLL(self.likelihood, self.gp)

    def get_posterior(
        self,
        X: Union[np.ndarray, torch.Tensor],
        calc_variance: bool = False,
        calc_covariance: bool = False,
        method: str = "cg",
        cg_tol: float = 0.1,
        max_cg_iter: int = 1000,
        preconditioner_size: int = 0,
        n_trace_samples: int = 50,
        root_decomposition_size: int = 50,
    ) -> MultivariateNormal:
        """
        Obtain the posterior distribution of the Gaussian process model
        for the given input sequences.

        This method computes the posterior distribution of the Gaussian process
        model for a set of input sequences. Depending on the specified options,
        it can compute the posterior mean, variance, or covariance matrix.

        Parameters
        ----------
        X : np.array or torch.Tensor of shape (n_sequences,)
            A vector or Tensor containing the sequences to predict.

        calc_variance : bool, optional (default=False)
            If True, computes the posterior variance in addition to the
            posterior mean. This is useful for uncertainty quantification
            in predictions.

        calc_covariance : bool, optional (default=False)
            If True, computes the posterior covariance matrix. This option
            provides the full covariance structure of the posterior distribution
            and overrides `calc_variance` if both are set to True.

        method : str, optional (default="cg")
            The method to use for posterior computation. Options are:
            - "cg": Conjugate gradient method for efficient computation.
            - "cholesky": Cholesky decomposition for exact computation.

        cg_tol : float, optional (default=0.1)
            The tolerance for the conjugate gradient method. Lower values
            result in higher precision but may increase computation time.

        max_cg_iter : int (default=1000)
            Maximum number of CG iterations to run.

        preconditioner_size : int, optional (default=0)
            The size of the preconditioner used to accelerate convergence
            in the conjugate gradient method. A value of 0 disables the
            preconditioner.

        n_trace_samples : int, optional (default=50)
            The number of samples used to estimate the trace of a matrix
            during computations. Increasing this value improves the accuracy
            of the trace estimation but increases computational cost.

        root_decomposition_size : int, optional (default=50)
            The size of the root decomposition used for approximating
            covariance matrices. Larger values improve accuracy but increase
            memory usage.

        Returns
        -------
        f : gpytorch.distributions.MultivariateNormal
            The posterior distribution of the Gaussian process model
            evaluated at the input sequences. The distribution includes
            the posterior mean and, optionally, the variance or covariance
            matrix depending on the specified options.

        Notes
        -----
        - When `calc_covariance` is True, the full covariance matrix is computed,
          which may be computationally expensive for large datasets.
        - When `calc_variance` is True, only the diagonal elements of the covariance
          matrix (variances) are computed, which is more efficient.
        - If neither `calc_variance` nor `calc_covariance` is True, only the posterior
          mean is computed.
        """
        self.set_evaluation_mode()
        X = self.encode(X)
        fast = self.get_fast_comp(method=method)

        with torch.inference_mode(), fast_computations(fast), max_preconditioner_size(
            preconditioner_size
        ), max_root_decomposition_size(root_decomposition_size), eval_cg_tolerance(
            cg_tol
        ), max_cg_iterations(max_cg_iter):
            if calc_covariance:
                f = self.gp(X)
            elif calc_variance:
                if method == "cg":
                    with fast_pred_var(num_probe_vectors=n_trace_samples):
                        f = self.gp(X)
                else:
                    with fast_pred_var(False):
                        f = self.gp(X)
            else:
                with skip_posterior_variances():
                    f = self.gp(X)
        return f

    def get_pred_dataframe(
        self,
        means: torch.Tensor,
        variances: Optional[torch.Tensor] = None,
        covariance: Optional[torch.Tensor] = None,
        labels: Optional[Any] = None,
    ) -> pd.DataFrame:
        if variances is not None and covariance is not None:
            msg = "Provide only variances or covariance"
            raise ValueError(msg)
        elif variances is None and covariance is not None:
            variances = covariance.diag()

        results = pd.DataFrame({"coef": to_numpy(means)}, index=labels)
        if variances is not None:
            results["stdev"] = to_numpy(torch.sqrt(variances))
            results["lower_ci"] = results["coef"] - 2 * results["stdev"]
            results["upper_ci"] = results["coef"] + 2 * results["stdev"]

        return results

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
        calc_variance: bool = False,
        method: str = "cg",
        cg_tol: float = 1e-4,
        max_cg_iter: int = 1000,
        preconditioner_size: int = 0,
        n_trace_samples: int = 50,
        root_decomposition_size: int = 50,
    ) -> pd.DataFrame:
        """
        Make phenotypic predictions using the Gaussian process model.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            A vector or tensor of shape (n_sequences,) containing the sequences
            for which predictions are to be made.

        calc_variance : bool, optional (default=False)
            If True, computes the posterior variance in addition to the posterior
            mean. This is useful for uncertainty quantification in predictions.

        method : str, optional (default="cg")
            The method to use for posterior computation. Options are:
            - "cg": Conjugate gradient method for efficient computation.
            - "cholesky": Cholesky decomposition for exact computation.

        cg_tol : float, optional (default=1e-4)
            The tolerance for the conjugate gradient method. Lower values result
            in higher precision but may increase computation time.

        max_cg_iter : int (default=1000)
            Maximum number of CG iterations to run.

        preconditioner_size : int, optional (default=0)
            The size of the preconditioner used to accelerate convergence in the
            conjugate gradient method. A value of 0 disables the preconditioner.

        n_trace_samples : int, optional (default=50)
            The number of samples used to estimate the trace of a matrix during
            computations. Increasing this value improves the accuracy of the trace
            estimation but increases computational cost.

        root_decomposition_size : int, optional (default=50)
            The size of the root decomposition used for approximating covariance
            matrices. Larger values improve accuracy but increase memory usage.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing phenotypic predictions for the input sequences.
            If `calc_variance=True`, the DataFrame includes posterior standard
            deviations and 95% credible interval bounds.
        """
        labels = X
        f = self.get_posterior(
            X,
            calc_variance=calc_variance,
            method=method,
            cg_tol=cg_tol,
            max_cg_iter=max_cg_iter,
            preconditioner_size=preconditioner_size,
            n_trace_samples=n_trace_samples,
            root_decomposition_size=root_decomposition_size,
        )
        fast = self.get_fast_comp(method)
        variances = None
        with torch.inference_mode(), fast_pred_var(
            fast, num_probe_vectors=n_trace_samples
        ), max_root_decomposition_size(root_decomposition_size), fast_computations(
            fast
        ), cg_tolerance(cg_tol), eval_cg_tolerance(cg_tol), max_preconditioner_size(
            preconditioner_size
        ):
            means = f.mean
            if calc_variance:
                variances = f.variance

        df = self.get_pred_dataframe(means=means, variances=variances, labels=labels)
        return df

    def make_contrasts(
        self,
        contrast_matrix: pd.DataFrame,
        calc_variance: bool = False,
        method: str = "cg",
        cg_tol: float = 1e-4,
        max_cg_iter: int = 1000,
        preconditioner_size: int = 0,
    ) -> pd.DataFrame:
        """
        Compute phenotypic contrasts across sets of genotypes
        using the Gaussian process model.

        Parameters
        ----------
        contrast_matrix : pd.DataFrame of shape (n_contrasts, n_sequences)
            A DataFrame where each row represents a linear combination
            of sequences encoded by `X`. The columns correspond to the
            sequences, and the values represent the coefficients for
            the linear combination.

        calc_variance : bool, optional (default=False)
            If True, computes the posterior (co)-variance in addition
            to the posterior mean.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the phenotypic contrasts for the
            desired sequences. If `calc_variance=True`, the DataFrame
            includes posterior standard deviations and 95% credible
            interval bounds.
        """
        X = contrast_matrix.columns.values
        contrasts = contrast_matrix.index.values
        B = self.get_tensor(contrast_matrix.values)
        fast = self.get_fast_comp(method=method)
        f = self.get_posterior(
            X,
            calc_covariance=calc_variance,
            method=method,
            cg_tol=cg_tol,
            max_cg_iter=max_cg_iter,
            preconditioner_size=preconditioner_size,
        )

        with torch.inference_mode(), fast_computations(fast), cg_tolerance(
            cg_tol
        ), eval_cg_tolerance(cg_tol), max_preconditioner_size(preconditioner_size):
            means = B @ f.mean
            variances = None
            if calc_variance:
                variances = (B @ f.lazy_covariance_matrix @ B.T).diag()

        results = self.get_pred_dataframe(means, variances, labels=contrasts)
        return results

    def predict_mut_effects(
        self,
        seq0: str,
        calc_variance: bool = False,
        method: str = "cg",
        cg_tol: float = 1e-4,
        max_cg_iter: int = 1000,
        preconditioner_size: int = 0,
    ) -> pd.DataFrame:
        """
        Predict the effects of single mutations on the phenotype
        using the Gaussian process model.

        This method computes the phenotypic effects of single mutations
        relative to a reference sequence. It uses the Gaussian process
        model to predict the effects and optionally computes the posterior
        variance for uncertainty quantification.

        Parameters
        ----------
        seq0 : str
            The reference sequence for which single mutation effects
            are to be predicted.

        calc_variance : bool, optional (default=False)
            If True, computes the posterior variance in addition to the
            posterior mean.

        method : str, optional (default="cg")
            The method to use for posterior computation. Options are:
            - "cg": Conjugate gradient method for efficient computation.
            - "cholesky": Cholesky decomposition for exact computation.

        cg_tol : float, optional (default=1e-4)
            The tolerance for the conjugate gradient method. Lower values
            result in higher precision but may increase computation time.

        max_cg_iter : int, optional (default=1000)
            Maximum number of CG iterations to run.

        preconditioner_size : int, optional (default=0)
            The size of the preconditioner used to accelerate convergence
            in the conjugate gradient method. A value of 0 disables the
            preconditioner.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the predicted effects of single mutations
            on the phenotype. If `calc_variance=True`, the DataFrame includes
            posterior standard deviations and 95% credible interval bounds.
        """
        contrast_matrix = get_mut_effs_contrast_matrix(seq0, self.alphabet_list)
        results = self.make_contrasts(
            contrast_matrix,
            calc_variance=calc_variance,
            method=method,
            cg_tol=cg_tol,
            max_cg_iter=max_cg_iter,
            preconditioner_size=preconditioner_size,
        )
        return results

    def predict_epistatic_coeffs(
        self,
        seq0: str,
        calc_variance: bool = False,
        method: str = "cg",
        cg_tol: float = 1e-4,
        max_cg_iter: int = 1000,
        preconditioner_size: int = 0,
    ) -> pd.DataFrame:
        """
        Compute epistatic coefficients across sets of genotypes
        using the Gaussian process model.

        Parameters
        ----------
        seq0 : str
            The reference sequence for which epistatic coefficients
            are to be predicted.

        calc_variance : bool, optional (default=False)
            If True, computes the posterior variance in addition to the
            posterior mean.

        method : str, optional (default="cg")
            The method to use for posterior computation. Options are:
            - "cg": Conjugate gradient method for efficient computation.
            - "cholesky": Cholesky decomposition for exact computation.

        cg_tol : float, optional (default=1e-4)
            The tolerance for the conjugate gradient method. Lower values
            result in higher precision but may increase computation time.

        max_cg_iter : int, optional (default=1000)
            Maximum number of CG iterations to run.

        preconditioner_size : int, optional (default=0)
            The size of the preconditioner used to accelerate convergence
            in the conjugate gradient method. A value of 0 disables the
            preconditioner.
        """
        contrast_matrix = get_epistatic_coeffs_contrast_matrix(seq0, self.alphabet_list)
        results = self.make_contrasts(
            contrast_matrix,
            calc_variance=calc_variance,
            method=method,
            cg_tol=cg_tol,
            max_cg_iter=max_cg_iter,
            preconditioner_size=preconditioner_size,
        )
        return results

    def get_prior(
        self, X: Union[np.ndarray, torch.Tensor], sigma2: float
    ) -> MultivariateNormal:
        X = self.encode(X)
        likelihood = FixedNoiseGaussianLikelihood(noise=sigma2 * torch.ones(X.shape[0]))
        gp = GPModel(self.kernel, None, None, likelihood, train_mean=self.train_mean)
        prior = gp.forward(X)
        return prior

    def simulate(
        self,
        X: Union[np.ndarray, torch.Tensor],
        n: int = 1,
        sigma2: float = 1e-4,
        method: str = "cholesky",
        root_decomposition_size: int = 50,
    ) -> torch.Tensor:
        """
        Sample random sequence-function relationships from the prior
        evaluated at the input sequences.

        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            Array or tensor of shape (n_sequences,) containing input sequences
            (or an already-encoded tensor accepted by self.encode).

        n : int, optional (default=1)
            Number of independent functions (landscapes) to sample from the prior.

        sigma2 : float, optional (default=1e-4)
            Observation noise variance to include in the prior (per-input variance).

        method : str, optional (default="cholesky")
            The method to use for sampling. Options are:
            - "lanczos": Use low rank Lanczos tridiagonalization approximation.
            - "cholesky": Cholesky decomposition for exact computation.

        root_decomposition_size : int, optional (default=50)
            Size of root decomposition used by approximate methods; larger values
            increase accuracy at higher memory cost.

        Returns
        -------
        torch.Tensor
            Samples from the prior with shape (n, n_sequences), where each row
            is one sampled function evaluated at the input sequences.
        """
        method = "cg" if method == "lanczos" else method
        fast = self.get_fast_comp(method)

        with fast_computations(
            covar_root_decomposition=fast
        ), max_root_decomposition_size(root_decomposition_size):
            prior = self.get_prior(X, sigma2=sigma2)
            v = torch.zeros(n)
            y = prior.sample(v.size())

        return y

    def simulate_dataset(
        self,
        X: Union[np.ndarray, torch.Tensor],
        sigma: float = 0,
        ptrain: float = 0.8,
        method: str = "cholesky",
        root_decomposition_size: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulate a dataset by sampling random sequence-function relationships
        from the prior and splitting the data into training and test sets.

        Parameters
        ----------
        X : torch.Tensor of shape (n_sequence, n_features)
            Tensor containing the one-hot encoding of the sequences
            to make predictions.

        sigma : float, optional (default=0)
            Standard deviation of the noise to add to the training data.
            If `sigma=0`, no noise is added.

        ptrain : float, optional (default=0.8)
            Proportion of the data to include in the training set.
            The remaining data will be used as the test set.

        Returns
        -------
        tuple
            A tuple containing:
            - train_x : torch.Tensor
            Training set input features.
            - train_y : torch.Tensor
            Training set target values.
            - test_x : torch.Tensor
            Test set input features.
            - test_y : torch.Tensor
            Test set target values.
            - train_y_var : torch.Tensor
            Variance of the training set target values.
        """
        f = self.simulate(
            X, n=1, method=method, root_decomposition_size=root_decomposition_size
        ).flatten()
        splits = split_training_test(X, f, y_var=None, ptrain=ptrain)
        train_x, train_f, test_x, test_f, train_y_var = splits
        if sigma > 0:
            train_y = torch.normal(mean=train_f, std=sigma)
            train_y_var = torch.full_like(train_y, sigma**2)

        return (train_x, train_y, test_x, test_f, train_y_var)

    def calc_kron_dot_map(
        self, x1: torch.Tensor, matrices: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the posterior mean for the Kronecker factorizable projection of the MAP.

        Parameters
        ----------
        x1 : torch.Tensor
            A tensor of shape (output_size,) containing one-hot encoded elements
            for which to compute the maximum a posteriori (MAP).

        matrices : list of torch.Tensor
            A list of Kronecker factors (matrices) to be used in the matrix-matrix
            multiplication.

        Returns
        -------
        torch.Tensor
            A tensor of shape (output_size,) representing the posterior mean
            for the Kronecker factorizable projection of the MAP.

        """
        if not isinstance(self.kernel, SiteProductKernel):
            msg = "calc_kron_dot_map can only be used for GPs with site factorizable kernels"
            raise ValueError(msg)

        if len(matrices) != self.l:
            msg = f"Number of matrices must be equal to sequence length {self.l}"
            raise ValueError(msg)

        output_sizes = [m.shape[0] for m in matrices]
        output_size = sum(output_sizes)
        if x1.shape[1] != output_size:
            msg = f"x1 has {x1.shape[0]} columns but expected {output_size}"
            raise ValueError(msg)

        vs = np.cumsum(output_sizes).astype(int)
        starts1 = np.append([0], vs[:-1])
        ends1 = vs
        x2 = self.X

        # Compute (P @ K)_{x1, x2} using Kronecker factorization
        site_kernels = self.kernel.get_site_kernels()
        PK = 1.0
        print([m.shape[1] for m in matrices])
        for p, (m, k) in enumerate(zip(matrices, site_kernels)):
            if m.shape[1] != k.shape[0]:
                msg = f"Incompatible size of matrices at position {p}: "
                msg += f"{m.shape[1], k.shape[0]}"
                raise ValueError(msg)
            s1, e1 = starts1[p], ends1[p]
            s2, e2 = self.kernel.starts[p], self.kernel.ends[p]
            x1p = x1[:, s1:e1].contiguous()
            x2pT = x2[:, s2:e2].T.contiguous()
            PK *= x1p @ m @ k.detach() @ x2pT

        post_mean = PK @ self.gp.prediction_strategy.mean_cache
        return post_mean

    def calc_gauge_fixed_theta(
        self, X: torch.Tensor, pi_lc: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the gauge-fixed theta parameter vector for the MAP estimate.

        This method calculates the gauge-fixed additive theta vector for the
        maximum a posteriori (MAP) estimate using the provided gauge-fixed
        additive theta vectors for each site in the sequence.

        Parameters
        ----------
        X : torch.Tensor
            A tensor containing the subsequences for which to compute the
            gauge-fixed parameter value.

        pi_lc : list of torch.Tensor
            A list of tensors representing the gauge-fixed additive theta vectors
            for each site in the sequence.

        Returns
        -------
        torch.Tensor
            A tensor representing the gauge-fixed additive theta vector for the
            MAP estimate.
        """
        if not isinstance(self.kernel, SiteProductKernel):
            msg = "calc_gauge_fixed_add_theta can only be used for GPs with site factorizable kernels"
            raise ValueError(msg)

        if len(pi_lc) != self.l:
            msg = f"Number of pi_lc tensors must be equal to sequence length {self.l}"
            raise ValueError(msg)

        P0s = [pi_p.unsqueeze(0) for pi_p in pi_lc]
        Ps = [torch.vstack([P0_p, torch.eye(P0_p.shape[1]) - P0_p]) for P0_p in P0s]
        x = self.get_tensor(get_one_hot_encoding(X, self.extended_alphabet_list))
        theta = self.calc_kron_dot_map(x, Ps)
        return theta

    def calc_gauge_fixed_add_theta(self, pi_lc: List[torch.Tensor]) -> pd.DataFrame:
        """
        Compute the gauge-fixed additive theta vector for the MAP estimate.

        This method calculates the gauge-fixed additive theta vector for the
        maximum a posteriori (MAP) estimate using the provided gauge-fixed
        additive theta vectors for each site in the sequence.

        Parameters
        ----------
        pi_lc : list of torch.Tensor
            A list of tensors representing the gauge-fixed additive theta vectors
            for each site in the sequence.

        Returns
        -------
        torch.Tensor
            A tensor representing the gauge-fixed additive theta vector for the
            MAP estimate, computed for all possible single-site mutations.
        """
        X = []
        names = []
        for p, alphabet_p in enumerate(self.alphabet_list):
            for a in alphabet_p:
                seq = ["*"] * self.l
                seq[p] = a
                X.append("".join(seq))
                names.append(f"{p}{a}")
        X = np.array(X)
        theta = self.calc_gauge_fixed_theta(X, pi_lc)
        theta = pd.DataFrame(
            {
                "theta": theta,
                "position": [int(n[0]) for n in names],
                "allele": [n[1] for n in names],
            },
            index=names,
        )
        return theta

    def calc_kron_quad_map(self, matrices: List[torch.Tensor]) -> float:
        """
        Compute the quadratic form of a Kronecker factorizable matrix A with
        the maximum a posteriori (MAP) estimate for the sequence-function map.

        This method calculates the quadratic form given by:
        .. math::
        f^T (\bigotimes_p^\ell A_p) f

        where :math:`\bigotimes` represents the Kronecker product.

        Parameters
        ----------
        matrices : list of torch.Tensor
            A list of Kronecker factors (matrices) representing the matrix A.

        Returns
        -------
        float
            The value of the quadratic form :math:`f^T A f`.

        Raises
        ------
        ValueError
            If the kernel is not a `SiteProductKernel` or if the dimensions
            of the provided matrices are incompatible with the kernel.
        """
        if not isinstance(self.kernel, SiteProductKernel):
            msg = "calc_kron_dot_map can only be used for GPs with site factorizable kernels"
            raise ValueError(msg)

        if len(matrices) != self.l:
            msg = f"Number of matrices must be equal to sequence length {self.l}"
            raise ValueError(msg)

        # Compute (K @ P @ K)_xx using Kronecker factorization
        site_kernels = self.kernel.get_site_kernels()
        KPK = 1.0
        for p, (m, k) in enumerate(zip(matrices, site_kernels)):
            if m.shape[1] != k.shape[0] or m.shape[0] != k.shape[0]:
                msg = f"Incompatible size of matrices at position {p}: "
                msg += f"{m.shape, k.shape}"
                raise ValueError(msg)
            x_p = self.kernel.select_site(self.X, p).contiguous()
            k_p = k.detach()
            kmk = k_p @ m @ k_p
            KPK *= x_p @ kmk @ x_p.T

        alpha = self.gp.prediction_strategy.mean_cache
        quad = torch.dot(alpha, KPK @ alpha).item()
        return quad


class GeneralizedEpiK(_Epik):
    def __init__(self, kernel: Any, likelihood: Any, **kwargs: Any) -> None:
        super(self).__init__(kernel, **kwargs)
        self.likelihood_function: Any = likelihood

    def get_likelihood(self, y_var: Any, train_noise: bool) -> Any:
        likelihood = self.likelihood_function(y_var, train_noise)

        if self.device is not None:
            likelihood = likelihood.cuda()
        return likelihood

    def define_negative_loss(self) -> None:
        self.calc_negative_loss = VariationalELBO(
            self.likelihood, self.gp, self.y.numel()
        )

    def define_model(self) -> None:
        self.gp: GeneralizedGPModel = GeneralizedGPModel(
            self.X,
            self.kernel,
            train_mean=self.train_mean,
            device=self.device,
            n_devices=self.n_devices,
        )
        if self.device is not None:
            self.gp: GeneralizedGPModel = self.gp.cuda()

    def predict(
        self, X: Any, nsamples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y_var = self.likelihood.second_noise * torch.ones(X.shape[0])
        likelihood = self.get_likelihood(y_var, train_noise=False)

        with torch.no_grad(), num_likelihood_samples(nsamples):
            phi = self.gp(X)
            y = likelihood(phi)
            yhat, y_var = y.mean.mean(0), y.variance.mean(0)
            phi = phi.mean.detach()
            return (phi, yhat, y_var)
