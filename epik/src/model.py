from tqdm import tqdm
import numpy as np
import torch
import gpytorch

from gpytorch.kernels.multi_device_kernel import MultiDeviceKernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.likelihoods.gaussian_likelihood import (FixedNoiseGaussianLikelihood,
                                                      GaussianLikelihood)
from scipy.stats.stats import pearsonr
from epik.src.utils import get_tensor, to_device


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, kernel, likelihood,
                 output_device=None, n_devices=None, train_mean=True):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        
        if train_mean:
            self.mean_module = gpytorch.means.ConstantMean()
        else:
            self.mean_module = gpytorch.means.ZeroMean()
        
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
                 dtype=torch.float32, train_mean=False,
                 alleles=None):
        self.kernel = kernel
        self.likelihood_type = likelihood_type
        self.output_device = output_device
        self.n_devices = n_devices
        self.dtype = dtype
        self.train_mean = train_mean
        self.alleles = alleles
    
    def set_likelihood(self, y_var=None):
        if self.likelihood_type == 'Gaussian':
            if y_var is not None:
                likelihood = FixedNoiseGaussianLikelihood(noise=self.get_tensor(y_var),
                                                          learn_additional_noise=True)
            else:
                likelihood = GaussianLikelihood()
        else:
            msg = 'Only Gaussian likelihood is allowed so far'
            raise ValueError(msg)
        
        self.likelihood = self.to_device(likelihood)
    
    def report_progress(self, pbar, loss, rho=None):
        if self.output_device is not None:
            loss = loss.cpu()
        report_dict = {'loss': '{:.3f}'.format(loss.detach().numpy())}
        if hasattr(self.model.covar_module, 'log_lda'):
            lambdas = self.model.covar_module.log_lda
            lambdas_text = ['{:.2f}'.format(l) for l in lambdas.detach().numpy()]
            lambdas_text = '[{}]'.format(', '.join(lambdas_text))
            report_dict['log(lambda)'] = lambdas_text
        
        if self.model.covar_module.lengthscale is not None:
            v = self.model.covar_module.lengthscale.detach().numpy()
            report_dict['lengthscale'] = '{:.2f}'.format(v[0][0])
        elif hasattr(self.model.covar_module, 'base_kernel'):
            if self.model.covar_module.base_kernel.lengthscale is not None:
                v = self.model.covar_module.base_kernel.lengthscale.detach().numpy()
                report_dict['lengthscale'] = '{:.2f}'.format(v[0][0])
        
        if rho is not None:
            report_dict['test rho'] = '{:.2f}'.format(rho)
        
        pbar.set_postfix(report_dict)
    
    def to_device(self, x):
        return(to_device(x, self.output_device))
    
    def get_tensor(self, ndarray):
        return(get_tensor(ndarray, dtype=self.dtype, device=self.output_device))
    
    def fit(self, X, y, y_var=None, n_iter=100, learning_rate=0.1):
        X, y = self.get_tensor(X), self.get_tensor(y)
        self.set_likelihood(y_var=y_var)
        self.model = self.to_device(GPModel(X, y, self.kernel, self.likelihood,
                                            output_device=self.output_device,
                                            n_devices=self.n_devices,
                                            train_mean=self.train_mean)) 

        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        pbar = tqdm(range(n_iter), desc='Iterations')
        for _ in pbar:
            try:
                optimizer.zero_grad()
                output = self.model(X)
                loss = -mll(output, y)
                loss.backward()
                optimizer.step()
                self.report_progress(pbar, loss)
            except KeyboardInterrupt:
                break
    
    def predict(self, pred_X):
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad(): #, gpytorch.settings.fast_pred_var():
            f_preds = self.model(self.get_tensor(pred_X))
        # TODO: error when asking for variance: f_preds.variance
        return(f_preds.mean)
        