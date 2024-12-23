import pandas as pd
import torch

from time import time
from tqdm import tqdm

from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.means import ZeroMean, ConstantMean
from gpytorch.kernels import MultiDeviceKernel
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from gpytorch.likelihoods import (FixedNoiseGaussianLikelihood, GaussianLikelihood)
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import (CholeskyVariationalDistribution,
                                  UnwhitenedVariationalStrategy)
from gpytorch.settings import (num_likelihood_samples, max_preconditioner_size, fast_pred_var,
                               skip_posterior_variances)

from epik.src.utils import (get_tensor, to_device, split_training_test, encode_seqs,
                            get_mut_effs_contrast_matrix, get_epistatic_coeffs_contrast_matrix)
 

class _GPModel(object):
    def init(self, kernel, train_mean, device, n_devices):
        if train_mean:
            self.mean_module = ConstantMean()
        else:
            self.mean_module = ZeroMean()
        
        self.covar_module = kernel
        if device is None:
            self.covar_module = kernel
        else:
            self.covar_module = MultiDeviceKernel(kernel,
                                                  device_ids=range(n_devices),
                                                  output_device=device)
            

class GPModel(ExactGP, _GPModel):
    def __init__(self, train_x, train_y, kernel, likelihood,
                 device=None, n_devices=None, train_mean=False):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.init(kernel, train_mean, device, n_devices)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return(MultivariateNormal(mean_x, covar_x))
        

class GeneralizedGPmodel(ApproximateGP, _GPModel):
    def __init__(self, train_x, kernel,
                 device=None, n_devices=None, train_mean=False):
        distribution = CholeskyVariationalDistribution(train_x.size(0))
        strategy = UnwhitenedVariationalStrategy(self, train_x, distribution,
                                                 learn_inducing_locations=False)
        super(GeneralizedGPmodel, self).__init__(strategy)
        self.init(kernel, train_mean, device, n_devices)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return(MultivariateNormal(mean_x, covar_x))


class _Epik(object):
    def __init__(self, kernel, device=None, n_devices=1,
                 train_mean=False, train_noise=False,
                 learning_rate=0.1, preconditioner_size=0,
                 track_progress=False):
        self.kernel = kernel
        self.device = device
        self.train_mean = train_mean
        self.learning_rate = learning_rate
        self.n_devices = n_devices
        self.preconditioner_size = preconditioner_size
        self.train_noise = train_noise
        self.track_progress = track_progress
        self.fit_time = 0
    
    def report_progress(self, pbar):
        if self.track_progress:
            report_dict = {'loss': '{:.3f}'.format(self.to_numpy(self.loss))}
            pbar.set_postfix(report_dict)
    
    def to_device(self, x):
        return(to_device(x, self.device))
    
    def to_numpy(self, v):
        if self.device is not None:
            v = v.cpu()
        return(v.detach().numpy())
    
    def get_tensor(self, ndarray):
        return(get_tensor(ndarray, device=self.device))
    
    def set_training_mode(self):
        self.model.train()
        self.likelihood.train()
        
    def set_evaluation_mode(self):
        self.model.eval()
        self.likelihood.eval()
    
    def define_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def set_data(self, X, y, y_var=None):
        '''
        Load data into model

        Parameters
        ----------
        X : torch.Tensor of shape (n_sequence, n_features)
            Tensor containing the one-hot encoding of the
            sequences to make predictions
        y : torch.Tensor of shape (n_sequence,)
            Tensor containing the phenotypic measurements for each
            sequence in `X`
        y_var : torch.Tensor of shape (n_sequence,) or None
            If `y_var=None` it is assumed that there is no uncertainty
            in the measurements. Otherwise, Tensor containing the
            variance of the measurements in `y`.

        '''
        self.X = self.get_tensor(X)
        self.y = self.get_tensor(y)
        self.y_var = y_var
        
        self.likelihood = self.get_likelihood(self.y_var, self.train_noise)
        self.define_model()
        self.define_optimizer()
        self.define_negative_loss()

    def calc_loss(self):
        return(-self.calc_negative_loss(self.model(self.X), self.y))
    
    def training_step(self, X, y):
        self.optimizer.zero_grad()
        self.loss = self.calc_loss()
        self.loss.backward()
        self.optimizer.step()
        self.params_history.append(self.model.state_dict())
        self.loss_history.append(self.loss.detach().item())
    
    def fit(self, n_iter=100):
        '''
        Function to optimize model hyperparamenters by maximizing 
        the marginal likelihood. This includes any kernel parameter, 
        as well as the optional mean and additional noise parameters. 
        
        Parameters
        ----------
        
        '''
        self.set_training_mode()
        
        pbar = range(n_iter)
        self.loss_history = []
        self.params_history = []
        
        with max_preconditioner_size(self.preconditioner_size):
            
            if n_iter > 1 and self.track_progress:
                pbar = tqdm(pbar, desc='Optimizing hyperparameters')
            
            t0 = time()
            for _ in pbar:
                self.training_step(self.X, self.y)
                if n_iter > 1:
                    self.report_progress(pbar)
                    
            self.fit_time = time() - t0
    
    @property
    def history(self):
        return(pd.DataFrame({'loss': self.loss_history}))
    
    def save(self, fpath):
        '''
        Store model parameters for future use

        Parameters
        ----------
        fpath : str
            File path for the file to store the parameters
            of the model
        '''
        torch.save(self.model.state_dict(), fpath)
        
    def load(self, fpath, **kwargs):
        '''
        Load model parameters from a file

        Parameters
        ----------
        fpath : str
            File path for the file with the stored model
            parameters
        '''
        self.model.load_state_dict(torch.load(fpath, **kwargs))


class EpiK(_Epik):
    '''
    Gaussian process regression model for inference of
    sequence-function relationships from experimental measurements
    using GPyTorch and KeOps backend.
    
    Parameters
    ----------
    kernel : epik.src.Kernel
        Instance of a kernel class characterizing the covariance
        between pairs of sequences to use for Gaussian process
        regression
        
    device : torch.device
        PyTorch device in which to run computation
             
    train_mean : bool (False)
        Option to optimize the mean function of the Gaussian Process. 
        By default it assumes a zero-mean
    
    train_noise : bool (False)
        Option to add unknown error to the Gaussian Process. By default
        it assumes that the provided error estimates are reliable
    
    learning_rate : float (0.1)
        Learning rate of the Adam optimizer used to optimize
        the hyperparameters through evidence maximization
    
    n_devices : int (1)    
        Number of GPUs to use for computation
        
    preconditioner_size : int (0)    
        Size of the preconditioner computed to accelerate
        conjugate gradient convergence. By default, no
        preconditioner is computed. 
    
    track_progress : bool (False)
        Option to show a progress bar for model fitting
        
    Returns
    -------
    model : epik.src.model.EpiK
        Instance of Gaussian Process model
    
    '''
    def get_likelihood(self, y_var=None, train_noise=False):
        if y_var is not None:
            likelihood = FixedNoiseGaussianLikelihood(noise=self.get_tensor(y_var),
                                                      learn_additional_noise=train_noise)
        else:
            likelihood = GaussianLikelihood()

        if self.device is not None:
            likelihood = likelihood.cuda()
        return(likelihood)
    
    def define_negative_loss(self):
        self.calc_negative_loss = ExactMarginalLogLikelihood(self.likelihood, self.model)
    
    def define_model(self):
        self.model = GPModel(self.X, self.y, self.kernel, self.likelihood,
                             train_mean=self.train_mean,
                             device=self.device,
                             n_devices=self.n_devices)
        if self.device is not None:
            self.model = self.model.cuda()
    
    def get_posterior(self, X, calc_variance=False, calc_covariance=False):
        
        self.set_evaluation_mode()
        X = self.get_tensor(X)
        
        with torch.no_grad(), max_preconditioner_size(self.preconditioner_size):
            if calc_covariance:
                f = self.model(X)
            elif calc_variance:
                with fast_pred_var(): 
                    f = self.model(X)
            else:
                with skip_posterior_variances():
                    f = self.model(X)
        return(f)
    
    def pred_to_df(self, res, calc_variance=False, labels=None):
        if calc_variance:
            m, x = res
            if len(x.shape) == 1:
                var = x
            else:
                var = x.diag()
            sd = self.to_numpy(torch.sqrt(var))
            m = self.to_numpy(m)
            result = pd.DataFrame({'coef': m, 'stderr': sd,
                                   'lower_ci': m - 2 * sd,
                                   'upper_ci': m + 2 * sd}, index=labels)
        else:
            result = pd.DataFrame({'coef': self.to_numpy(res)}, index=labels)
        return(result)
    
    def predict(self, X, calc_variance=False, labels=None):
        '''
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
        
        '''
        t0 = time()
        self.set_evaluation_mode()
        X = self.get_tensor(X)
        f = self.get_posterior(X, calc_variance=calc_variance)
        
        res = (f.mean, f.variance) if calc_variance else f.mean
        df = self.pred_to_df(res, calc_variance=calc_variance,
                             labels=labels)
        self.pred_time = time() - t0
        return(df)
    
    def make_contrasts(self, contrast_matrix, X,
                       calc_variance=False):
        '''
        Function to make contrasts of phenotypes across sets of genotypes
        under the Gaussian process model
        
        Parameters
        ----------
        contrast_matrix : torch.Tensor of shape (n_contrasts, n_sequences)
            Tensor containing the linear combination of sequences in
            encoded by `X` to compute posterior distribution of 
            the contrasts. 
        
        X : torch.Tensor of shape (n_sequences, n_features)
            Tensor containing the one-hot encoding of the
            sequences to make predictions
            
        calc_variance : bool (False)
            Option to compute the posterior (co)-variance in addition
            to the posterior mean reported by default

        Returns
        -------
        output : torch.Tensor or (torch.Tensor, torch.Tensor)
-            Tensor containing phenotypic predictions at the
-            desired sequences. If `calc_variance=True`, a second 
-            Tensor containing the covariance matrix of the posterior
-            of the contrasts will be returned.
        '''
        t0 = time()
        self.set_evaluation_mode()

        B = self.get_tensor(contrast_matrix)
        f = self.get_posterior(X, calc_covariance=calc_variance)

        res = B @ f.mean
        if calc_variance:
            res = res, (B @ f.covariance_matrix @ B.T)
        
        self.contrast_time = time() - t0
        return(res)

    def predict_contrasts(self, contrast_matrix, alleles,
                          calc_variance=False, max_size=100):
        n = contrast_matrix.shape[0]
        n_chunks = int(n / max_size) + 1
        results = []
        for i in tqdm(range(n_chunks), total=n_chunks):
            df = contrast_matrix.iloc[i*max_size:(i+1)*max_size, :]
            seqs, labels = df.columns, df.index
            X = encode_seqs(seqs, alphabet=alleles)
            C = torch.Tensor(df.values)
            res = self.make_contrasts(C, X, calc_variance=calc_variance)
            result = self.pred_to_df(res, calc_variance=calc_variance, labels=labels)
            results.append(result)
        results = pd.concat(results)
        return(results)
    
    def predict_mut_effects(self, seq0, alleles, calc_variance=False, max_size=100):
        contrast_matrix = get_mut_effs_contrast_matrix(seq0, alleles)
        return(self.predict_contrasts(contrast_matrix, alleles,
                                      calc_variance=calc_variance, max_size=max_size))
    
    def predict_epistatic_coeffs(self, seq0, alleles, calc_variance=False, max_size=100):
        contrast_matrix = get_epistatic_coeffs_contrast_matrix(seq0, alleles)
        return(self.predict_contrasts(contrast_matrix, alleles,
                                      calc_variance=calc_variance, max_size=max_size))
    
    def get_prior(self, X, sigma2):
        self.X = X
        likelihood = FixedNoiseGaussianLikelihood(noise=sigma2 * torch.ones(X.shape[0]))
        model = self.to_device(GPModel(None, None, self.kernel, likelihood,
                                       train_mean=self.train_mean,
                                       device=self.device,
                                       n_devices=self.n_devices))
        prior = model.forward(X)
        return(prior)
    
    def simulate(self, X, n=1, sigma2=1e-4):
        '''
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
        '''
        
        prior = self.get_prior(X, sigma2=sigma2)
        v = torch.zeros(n)
        with torch.no_grad(), max_preconditioner_size(self.preconditioner_size):
            y = prior.rsample(v.size())
        return(y)
    
    def simulate_dataset(self, X, sigma=0, ptrain=0.8):
        y_true = self.simulate(X, n=1).flatten()
        y_true = y_true / y_true.std()
        
        splits = split_training_test(X, y_true, y_var=None, ptrain=ptrain)
        train_x, train_y, test_x, test_y, train_y_var = splits
        if sigma > 0:
            train_y = torch.normal(train_y, sigma)
            train_y_var = torch.full_like(train_y, sigma**2)
            
        return(train_x, train_y, test_x, test_y, train_y_var)


class GeneralizedEpiK(_Epik):
    def __init__(self, kernel, likelihood, **kwargs):
        super(self).__init__(kernel, **kwargs)
        self.likelihood_function = likelihood
        
    def get_likelihood(self, y_var, train_noise):
        likelihood = self.likelihood_function(y_var, train_noise)

        if self.device is not None:
            likelihood = likelihood.cuda()
        return(likelihood)
    
    def define_negative_loss(self):
        self.calc_negative_loss = VariationalELBO(self.likelihood, self.model, self.y.numel())
    
    def define_model(self):
        self.model = GeneralizedGPmodel(self.X, self.kernel,
                                        train_mean=self.train_mean,
                                        device=self.device,
                                        n_devices=self.n_devices)
        if self.device is not None:
            self.model = self.model.cuda()
    
    def predict(self, X, nsamples=100):
        y_var = self.likelihood.second_noise * torch.ones(X.shape[0])
        likelihood = self.get_likelihood(y_var, train_noise=False)
        
        with torch.no_grad(), num_likelihood_samples(nsamples):
            phi = self.model(X)
            y = likelihood(phi)
            yhat, y_var = y.mean.mean(0), y.variance.mean(0)
            phi = phi.mean.detach()
            return(phi, yhat, y_var)
        