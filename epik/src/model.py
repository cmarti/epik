import numpy as np
import gc
import torch
import gpytorch

from time import time
from tqdm import tqdm

from gpytorch.means import ZeroMean, ConstantMean
from gpytorch.kernels.multi_device_kernel import MultiDeviceKernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.likelihoods.gaussian_likelihood import (FixedNoiseGaussianLikelihood,
                                                      GaussianLikelihood)

from epik.src.utils import get_tensor, to_device, get_gpu_memory


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, kernel, likelihood,
                 output_device=None, n_devices=None, train_mean=False):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        
        if train_mean:
            self.mean_module = ConstantMean()
        else:
            self.mean_module = ZeroMean()
        
        if output_device is None:
            self.covar_module = kernel
        else:
            self.covar_module = MultiDeviceKernel(kernel,
                                                  device_ids=range(n_devices),
                                                  output_device=output_device)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class EpiK(object):
    def __init__(self, kernel, likelihood_type='Gaussian',
                 output_device=None, n_devices=1,
                 train_mean=False, train_noise=False,
                 partition_size=0, learning_rate=0.1,
                 preconditioner_size=0, dtype=torch.float32,
                 track_progress=True):
        self.kernel = kernel
        self.likelihood_type = likelihood_type
        self.output_device = output_device
        self.train_mean = train_mean
        self.learning_rate = learning_rate
        self.n_devices = n_devices
        self.dtype = dtype
        self.partition_size = partition_size
        self.preconditioner_size = preconditioner_size
        self.train_noise = train_noise
        self.track_progress = track_progress
    
    def set_likelihood(self, y_var=None):
        if self.likelihood_type == 'Gaussian':
            if y_var is not None:
                likelihood = FixedNoiseGaussianLikelihood(noise=self.get_tensor(y_var),
                                                          learn_additional_noise=self.train_noise)
            else:
                likelihood = GaussianLikelihood()
        else:
            msg = 'Only Gaussian likelihood is allowed so far'
            raise ValueError(msg)
        
        self.likelihood = self.to_device(likelihood)
    
    def get_mem_usage(self):
        return(get_gpu_memory(self.output_device))
    
    def report_progress(self, pbar):
        if self.track_progress:
            report_dict = {'loss': '{:.3f}'.format(self.to_numpy(self.loss)),
                           'mem': self.get_mem_usage()}
            pbar.set_postfix(report_dict)
    
    def to_device(self, x):
        return(to_device(x, self.output_device))
    
    def to_numpy(self, v):
        if self.output_device is not None:
            v = v.cpu()
        return(v.detach().numpy())
    
    def get_tensor(self, ndarray):
        return(get_tensor(ndarray, dtype=self.dtype, device=self.output_device))
    
    def training_step(self, X, y):
        params = self.kernel.get_params()
        self.optimizer.zero_grad()
        self.loss = -self.calc_mll(self.model(X), y)
        self.loss.backward()
        self.optimizer.step()
        params['loss'] = self.loss.detach().item()
        self.loss_history.append(params)
        
    def set_training_mode(self):
        self.model.train()
        self.likelihood.train()
        
    def set_evaluation_mode(self):
        self.model.eval()
        self.likelihood.eval()
    
    def set_partition_size(self, partition_size=None):
        if partition_size is None:
            partition_size = self.partition_size
        return(gpytorch.beta_features.checkpoint_kernel(partition_size))
    
    def set_preconditioner_size(self, preconditioner_size=None):
        if preconditioner_size is None:
            preconditioner_size = self.preconditioner_size
        return(gpytorch.settings.max_preconditioner_size(preconditioner_size))
    
    def define_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def define_loss(self):
        self.calc_mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
    
    def define_model(self):
        self.model = self.to_device(GPModel(self.X, self.y, self.kernel, self.likelihood,
                                            train_mean=self.train_mean,
                                            output_device=self.output_device,
                                            n_devices=self.n_devices))
    
    def get_gp_mean(self):
        return(self.to_numpy(self.model.mean_module.constant)) 
    
    def set_data(self, X, y, y_var=None):
        self.X = self.get_tensor(X)
        self.y = self.get_tensor(y)
        self.y_var = y_var
        
        self.set_likelihood(y_var=self.y_var)
        self.define_model()
        self.define_optimizer()
        self.define_loss()
    
    def fit(self, n_iter=100):
        self.set_training_mode()
        
        pbar = range(n_iter)
        self.loss_history = []
        
        with self.set_partition_size(), self.set_preconditioner_size():
            
            if n_iter > 1 and self.track_progress:
                pbar = tqdm(pbar, desc='Maximizing Marginal Likelihood')
            
            t0 = time()
            
            for _ in pbar:
                self.training_step(self.X, self.y)
                if n_iter > 1:
                    self.report_progress(pbar)
                    
            self.fit_time = time() - t0
    
    def predict(self, pred_X, calc_variance=False):
        if calc_variance:
            msg = 'Variance calculation not implemented yet'
            raise ValueError(msg)
        
        t0 = time()
        
        self.set_evaluation_mode()
        pred_X = self.get_tensor(pred_X)
        
        with torch.no_grad(), self.set_preconditioner_size(), self.set_partition_size(): #, , gpytorch.settings.fast_pred_var():
            f_preds = self.model(pred_X).mean

        self.pred_time = time() - t0
        return(f_preds)
    
    def get_prior(self, X, sigma2):
        likelihood = FixedNoiseGaussianLikelihood(noise=sigma2 * torch.ones(X.shape[0]))
        model = self.to_device(GPModel(None, None, self.kernel, likelihood,
                                       train_mean=self.train_mean,
                                       output_device=self.output_device,
                                       n_devices=self.n_devices))
        prior = model.forward(X)
        return(prior)
    
    def sample(self, X, n=1, sigma2=1e-4):
        prior = self.get_prior(X, sigma2=sigma2)
        v = torch.zeros(n)
        with torch.no_grad(), self.set_preconditioner_size(), self.set_partition_size():
            y = prior.rsample(v.size())
        return(y)
