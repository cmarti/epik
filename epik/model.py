import pandas as pd
import torch

from copy import deepcopy
from time import time
from tqdm import tqdm

from torch.optim import Adam, SGD
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.means import ZeroMean, ConstantMean
from gpytorch.mlls import MarginalLogLikelihood, VariationalELBO
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, _GaussianLikelihoodBase
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    UnwhitenedVariationalStrategy,
)
from gpytorch.settings import (
    num_likelihood_samples,
    max_preconditioner_size,
    max_lanczos_quadrature_iterations,
    fast_pred_var,
    skip_posterior_variances,
    cg_tolerance,
    num_trace_samples,
)

from epik.utils import (
    get_tensor,
    to_numpy,
    split_training_test,
    encode_seqs,
    get_mut_effs_contrast_matrix,
    get_epistatic_coeffs_contrast_matrix,
)


class ExactMLL(MarginalLogLikelihood):
    def __init__(self, likelihood, model):
        if not isinstance(likelihood, _GaussianLikelihoodBase):
            raise RuntimeError("Likelihood must be Gaussian for exact inference")
        super(ExactMLL, self).__init__(likelihood, model)

    def _add_other_terms(self, res, params):
        # Add additional terms (SGPR / learned inducing points, heteroskedastic likelihood models)
        for added_loss_term in self.model.added_loss_terms():
            res = res.add(added_loss_term.loss(*params))

        # Add log probs of priors on the (functions of) parameters
        res_ndim = res.ndim
        for name, module, prior, closure, _ in self.model.named_priors():
            prior_term = prior.log_prob(closure(module))
            res.add_(prior_term.view(*prior_term.shape[:res_ndim], -1).sum(dim=-1))

        return res

    def forward(self, function_dist, target, *params):
        if not isinstance(function_dist, MultivariateNormal):
            raise RuntimeError(
                "ExactMarginalLogLikelihood can only operate on Gaussian random variables"
            )

        # Get the log prob of the marginal distribution
        output = self.likelihood(function_dist, *params)
        res = output.log_prob(target)
        res = self._add_other_terms(res, params)
        return res


class GPModel(ExactGP):
    def __init__(self, train_x, train_y, kernel, likelihood, train_mean=False):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean() if train_mean else ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class GeneralizedGPModel(ApproximateGP):
    def __init__(self, train_x, kernel, train_mean=False):
        distribution = CholeskyVariationalDistribution(train_x.size(0))
        strategy = UnwhitenedVariationalStrategy(
            self, train_x, distribution, learn_inducing_locations=False
        )
        super(GeneralizedGPModel, self).__init__(strategy)
        self.mean_module = ConstantMean() if train_mean else ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class _Epik(object):
    def __init__(
        self,
        kernel,
        device="cpu",
        train_mean=False,
        train_noise=False,
        preconditioner_size=0,
        cg_tol=1.0,
        num_trace_samples=50,
        max_n_lanczos_iterations=50,
        track_progress=False,
    ):
        self.kernel = kernel
        self.device = device
        self.train_mean = train_mean
        self.train_noise = train_noise

        self.preconditioner_size = preconditioner_size
        self.cg_tol = cg_tol
        self.num_trace_samples = num_trace_samples
        self.max_n_lanczos_iterations = max_n_lanczos_iterations
        self.track_progress = track_progress
        self.fit_time = 0
        self.training_history = []
        self.params_history = []

    def report_progress(self, pbar):
        if self.track_progress:
            allocated = torch.cuda.memory_allocated(device="cuda") / 1e6
            reserved = torch.cuda.memory_reserved(device="cuda") / 1e6
            report_dict = {
                "MLL": f"{self.mll:.3f}",
                "Mem(alloc/res)": f"{allocated:.2f}/{reserved:.2f}MB",
            }
            if hasattr(self, "scheduler"):
                report_dict["LR"] = f"{self.optimizer.param_groups[0]['lr']:.4f}"

            pbar.set_postfix(report_dict)

    def get_tensor(self, ndarray):
        return get_tensor(ndarray, device=self.device)

    def set_training_mode(self):
        self.gp.train()
        self.likelihood.train()

    def set_evaluation_mode(self):
        self.gp.eval()
        self.likelihood.eval()

    def set_data(self, X, y, y_var=None):
        """
        Set the training data for the model.

        Parameters
        ----------
        X : torch.Tensor
            A tensor of shape (n_sequences, n_features) containing the 
            one-hot encoded input sequences.

        y : torch.Tensor
            A tensor of shape (n_sequences,) containing the phenotypic 
            measurements corresponding to each sequence in `X`.

        y_var : torch.Tensor, optional
            A tensor of shape (n_sequences,) representing the variance 
            of the measurements in `y`. If `None`, it is assumed that 
            there is no uncertainty in the measurements.
        """

        self.X = self.get_tensor(X)
        self.y = self.get_tensor(y)
        self.y_var = (
            torch.zeros_like(self.y) if y_var is None else self.get_tensor(y_var)
        )

        self.define_likelihood()
        self.define_gp()

    def calc_mll(self):
        with max_preconditioner_size(self.preconditioner_size), cg_tolerance(
            self.cg_tol
        ), num_trace_samples(self.num_trace_samples), max_lanczos_quadrature_iterations(
            self.max_n_lanczos_iterations
        ):
            return self.mll_layer(self.gp(self.X), self.y)

    def training_step(self):
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        neg_mll = -self.calc_mll()
        neg_mll.backward()
        self.optimizer.step()

        self.params = self.gp.state_dict()
        self.mll = -neg_mll.detach().item()
        self.params_history.append(self.params)
        self.training_history.append(self.mll)

        if not hasattr(self, "max_mll") or self.mll > self.max_mll:
            self.max_mll = self.mll
            self.max_params = deepcopy(self.params)

    def fit(self, n_iter=100, learning_rate=0.1, optimizer="Adam"):
        """
        Optimize model hyperparameters by maximizing the marginal likelihood.

        This process adjusts kernel parameters, as well as optional mean and 
        additional noise parameters, to improve the model's performance.

        Parameters
        ----------
        n_iter : int, optional (default=100)
            Number of iterations for the optimization process.

        learning_rate : float, optional (default=0.1)
            Learning rate for the optimizer.

        optimizer : str, optional (default="Adam")
            Optimizer to use for hyperparameter optimization. Options are 
            "Adam" or "SGD".

        Raises
        ------
        ValueError
            If the specified optimizer is not recognized.
        """
        if optimizer == "Adam":
            self.optimizer = Adam(self.gp.parameters(), lr=learning_rate)
        elif optimizer == "SGD":
            self.optimizer = SGD(self.gp.parameters(), lr=learning_rate)
        else:
            raise ValueError("Optimizer {} not recognized".format(optimizer))
        
        self.set_training_mode()

        t0 = time()
        pbar = range(n_iter)
        if n_iter > 1 and self.track_progress:
            pbar = tqdm(pbar, desc="Optimizing hyperparameters")

        for _ in pbar:
            try:
                self.training_step()
            except RuntimeError:
                self.kernel.use_keops = True
                self.training_step()

            if n_iter > 1:
                self.report_progress(pbar)

        self.fit_time = time() - t0

    @property
    def history(self):
        return pd.DataFrame({"mll": self.training_history})

    def get_params(self):
        return self.gp.state_dict()

    def save(self, fpath):
        """
        Save the model parameters to a file for future use.

        Parameters
        ----------
        fpath : str
            The file path where the model parameters will be saved.
        """
        torch.save(self.gp.state_dict(), fpath)

    def set_params(self, params):
        self.gp.load_state_dict(params)

    def load(self, fpath, **kwargs):
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
    def define_likelihood(self):
        self.likelihood = FixedNoiseGaussianLikelihood(
            noise=self.y_var, learn_additional_noise=self.train_noise
        )

        if self.device == "cuda":
            self.likelihood = self.likelihood.cuda()

    def get_gp(self, likelihood, x=None, y=None):
        gp = GPModel(x, y, self.kernel, likelihood, train_mean=self.train_mean)

        if self.device == "cuda":
            gp = gp.cuda()

        return gp

    def define_gp(self):
        self.gp = self.get_gp(self.likelihood, self.X, self.y)
        self.mll_layer = ExactMLL(self.likelihood, self.gp)

    def get_posterior(self, X, calc_variance=False, calc_covariance=False):
        self.set_evaluation_mode()
        X = self.get_tensor(X)

        with torch.no_grad(), max_preconditioner_size(self.preconditioner_size):
            if calc_covariance:
                f = self.gp(X)
            elif calc_variance:
                with fast_pred_var():
                    f = self.gp(X)
            else:
                with skip_posterior_variances():
                    f = self.gp(X)
        return f

    def pred_to_df(self, res, calc_variance=False, labels=None):
        if calc_variance:
            m, x = res
            if len(x.shape) == 1:
                var = x
            else:
                var = x.diag()
            sd = to_numpy(torch.sqrt(var))
            m = to_numpy(m)
            result = pd.DataFrame(
                {
                    "coef": m,
                    "stderr": sd,
                    "lower_ci": m - 2 * sd,
                    "upper_ci": m + 2 * sd,
                },
                index=labels,
            )
        else:
            result = pd.DataFrame({"coef": to_numpy(res)}, index=labels)
        return result

    def predict(self, X, calc_variance=False, labels=None):
        """
        Function to make phenotypic predictions under the
        Gaussian process model

        Parameters
        ----------
        X : torch.Tensor of shape (n_sequences, n_features)
            Tensor containing the one-hot encoding of the
            sequences to make predictions

        calc_variance : bool (False)
            Option to compute the posterior variance in addition
            to the posterior mean reported by default

        labels : array-like of shape (n_sequences,) or None
            Sequence labels to use as rownames in the output
            pd.DataFrame

        Returns
        -------
        output : pd.DataFrame of shape (n_sequences, 1 or 4)
            DataFrame containing phenotypic predictions at the
            desired sequences. If `calc_variance=True`, posterior
            standard deviations and 95% credible interval
            bounds are added

        """
        t0 = time()
        self.set_evaluation_mode()
        X = self.get_tensor(X)
        f = self.get_posterior(X, calc_variance=calc_variance)

        res = (f.mean, f.variance) if calc_variance else f.mean
        df = self.pred_to_df(res, calc_variance=calc_variance, labels=labels)
        self.pred_time = time() - t0
        return df

    def make_contrasts(self, contrast_matrix, X, calc_variance=False):
        """
        Compute phenotypic contrasts across sets of genotypes
        using the Gaussian process model.

        Parameters
        ----------
        contrast_matrix : torch.Tensor of shape (n_contrasts, n_sequences)
            A tensor representing the linear combinations of sequences 
            encoded by `X` to compute the posterior distribution of 
            the contrasts.

        X : torch.Tensor of shape (n_sequences, n_features)
            A tensor containing the one-hot encoded sequences for 
            which predictions are to be made.

        calc_variance : bool, optional (default=False)
            If True, computes the posterior (co)-variance in addition 
            to the posterior mean.

        Returns
        -------
        output : torch.Tensor or tuple of torch.Tensor
            If `calc_variance=False`, returns a tensor containing the 
            phenotypic predictions for the desired sequences. 
            If `calc_variance=True`, returns a tuple where the first 
            element is the phenotypic predictions and the second element 
            is the covariance matrix of the posterior contrasts.
        """
        t0 = time()
        self.set_evaluation_mode()

        B = self.get_tensor(contrast_matrix)
        f = self.get_posterior(X, calc_covariance=calc_variance)

        res = B @ f.mean
        if calc_variance:
            res = res, (B @ f.covariance_matrix @ B.T)

        self.contrast_time = time() - t0
        return res

    def predict_contrasts(
        self, contrast_matrix, alleles, calc_variance=False, max_size=100
    ):
        n = contrast_matrix.shape[0]
        n_chunks = int(n / max_size) + 1
        results = []
        chunks = range(n_chunks)
        if self.track_progress:
            chunks = tqdm(chunks, total=n_chunks)
        for i in chunks:
            df = contrast_matrix.iloc[i * max_size : (i + 1) * max_size, :]
            seqs, labels = df.columns, df.index
            X = encode_seqs(seqs, alphabet=alleles)
            C = torch.Tensor(df.values)
            res = self.make_contrasts(C, X, calc_variance=calc_variance)
            result = self.pred_to_df(res, calc_variance=calc_variance, labels=labels)
            results.append(result)
        results = pd.concat(results)
        return results

    def predict_mut_effects(self, seq0, alleles, calc_variance=False, max_size=100):
        """
        Predict the effects of single mutations on the phenotype
        using the Gaussian process model.

        Parameters
        ----------
        seq0 : str
            The reference sequence for which single mutation effects
            are to be predicted.

        alleles : list of str
            A list of possible alleles for each position in the sequence.

        calc_variance : bool, optional (default=False)
            If True, computes the posterior variance in addition to the
            posterior mean.

        max_size : int, optional (default=100)
            The maximum number of contrasts to process in a single batch.
            Larger values may increase memory usage.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the predicted effects of single mutations
            on the phenotype. If `calc_variance=True`, the DataFrame includes
            posterior standard deviations and 95% credible interval bounds.
        """
        contrast_matrix = get_mut_effs_contrast_matrix(seq0, alleles)
        return self.predict_contrasts(
            contrast_matrix, alleles, calc_variance=calc_variance, max_size=max_size
        )

    def predict_epistatic_coeffs(
        self, seq0, alleles, calc_variance=False, max_size=100
    ):
        """
        Compute epistatic coefficients across sets of genotypes
        using the Gaussian process model.

        Parameters
        ----------
        seq0 : str
            The reference sequence for which epistatic coefficients
            are to be predicted.

        alleles : list of str
            A list of possible alleles for each position in the sequence.

        calc_variance : bool, optional (default=False)
            If True, computes the posterior (co)-variance in addition
            to the posterior mean.

        max_size : int, optional (default=100)
            The maximum number of contrasts to process in a single batch.
            Larger values may increase memory usage.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the predicted epistatic coefficients
            for the desired sequences. If `calc_variance=True`, the DataFrame
            includes posterior standard deviations and 95% credible interval
            bounds.
        """
        contrast_matrix = get_epistatic_coeffs_contrast_matrix(seq0, alleles)
        return self.predict_contrasts(
            contrast_matrix, alleles, calc_variance=calc_variance, max_size=max_size
        )

    def get_prior(self, X, sigma2):
        likelihood = FixedNoiseGaussianLikelihood(noise=sigma2 * torch.ones(X.shape[0]))
        gp = GPModel(None, None, self.kernel, likelihood, train_mean=self.train_mean)
        prior = gp.forward(X)
        return prior

    def simulate(self, X, n=1, sigma2=1e-4):
        """
        Sample random sequence-function relationships from the prior
        evaluated at the input sequences

        Parameters
        ----------
        X : torch.Tensor of shape (n_sequence, n_features)
            Tensor containing the one-hot encoding of the
            sequences to make predictions
        n : int (1)
            Number of sequence-function relationships to
            sample from the prior
        sigma2 : float (1e-4)
            Additional random noise to add to the
            simulated landscapes

        Returns
        -------
        y : torch.Tensor of shape (n_sequences, n)
            Tensor containing the simulated landscapes
            evaluated in the input sequences
        """

        with torch.no_grad(), max_preconditioner_size(self.preconditioner_size):
            prior = self.get_prior(X, sigma2=sigma2)
            v = torch.zeros(n)
            y = prior.rsample(v.size())

        return y

    def simulate_dataset(self, X, sigma=0, ptrain=0.8):
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
        y_true = self.simulate(X, n=1).flatten()
        y_true = y_true / y_true.std()

        splits = split_training_test(X, y_true, y_var=None, ptrain=ptrain)
        train_x, train_y, test_x, test_y, train_y_var = splits
        if sigma > 0:
            train_y = torch.normal(train_y, sigma)
            train_y_var = torch.full_like(train_y, sigma**2)

        return (train_x, train_y, test_x, test_y, train_y_var)


class GeneralizedEpiK(_Epik):
    def __init__(self, kernel, likelihood, **kwargs):
        super(self).__init__(kernel, **kwargs)
        self.likelihood_function = likelihood

    def get_likelihood(self, y_var, train_noise):
        likelihood = self.likelihood_function(y_var, train_noise)

        if self.device is not None:
            likelihood = likelihood.cuda()
        return likelihood

    def define_negative_loss(self):
        self.calc_negative_loss = VariationalELBO(
            self.likelihood, self.gp, self.y.numel()
        )

    def define_model(self):
        self.gp = GeneralizedGPModel(
            self.X,
            self.kernel,
            train_mean=self.train_mean,
            device=self.device,
            n_devices=self.n_devices,
        )
        if self.device is not None:
            self.gp = self.gp.cuda()

    def predict(self, X, nsamples=100):
        y_var = self.likelihood.second_noise * torch.ones(X.shape[0])
        likelihood = self.get_likelihood(y_var, train_noise=False)

        with torch.no_grad(), num_likelihood_samples(nsamples):
            phi = self.gp(X)
            y = likelihood(phi)
            yhat, y_var = y.mean.mean(0), y.variance.mean(0)
            phi = phi.mean.detach()
            return (phi, yhat, y_var)
