from tqdm import tqdm
import numpy as np
import torch
import gpytorch

from gpytorch.kernels.multi_device_kernel import MultiDeviceKernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.likelihoods.gaussian_likelihood import (FixedNoiseGaussianLikelihood,
                                                      GaussianLikelihood)


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, kernel, likelihood,
                 output_device=None, n_devices=None, train_mean=False):
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
                 dtype=torch.float32, train_mean=False):
        self.kernel = kernel
        self.likelihood_type = likelihood_type
        self.output_device = output_device
        self.n_devices = n_devices
        self.dtype = dtype
        self.train_mean = train_mean
    
    def seq_to_one_hot(self, X):
        m = np.array([[a for a in x] for x in X])
        onehot = []
        for i in range(m.shape[1]):
            c = m[:, i]
            alleles = np.unique(c)
            for allele in alleles:
                onehot.append(self.get_tensor(c == allele))
        onehot = torch.stack(onehot, 1)
        return(onehot)

    def to_device(self, tensor):
        if self.output_device is not None:
            tensor = tensor.to(self.output_device)
        return(tensor)

    def get_tensor(self, ndarray):
        if not torch.is_tensor(ndarray):
            ndarray = torch.tensor(ndarray, dtype=self.dtype)
        return(self.to_device(ndarray))
    
    def set_likelihood(self, y_var=None):
        if self.likelihood_type == 'Gaussian':
            if y_var is not None:
                likelihood = FixedNoiseGaussianLikelihood(noise=self.get_tensor(y_var),
                                                          learn_additional_noise=False)
            else:
                likelihood = GaussianLikelihood()
        else:
            msg = 'Only Gaussian likelihood is allowed so far'
            raise ValueError(msg)
        
        self.likelihood = self.to_device(likelihood)
    
    def report_progress(self, pbar, loss):
        if self.output_device is not None:
            loss = loss.cpu()
        report_dict = {'loss': '{:.3f}'.format(loss.detach().numpy())}
        if hasattr(self.model.covar_module, 'log_lda'):
            lambdas = self.model.covar_module.log_lda
            lambdas_text = ['{:.2f}'.format(np.exp(l)) for l in lambdas.detach().numpy()]
            lambdas_text = '[{}]'.format(', '.join(lambdas_text))
            report_dict['lambdas'] = lambdas_text
        
        if self.model.covar_module.lengthscale is not None:
            v = self.model.covar_module.lengthscale.detach().numpy()
            report_dict['lengthscale'] = '{:.2f}'.format(v[0][0])
        elif hasattr(self.model.covar_module, 'base_kernel'):
            if self.model.covar_module.base_kernel.lengthscale is not None:
                v = self.model.covar_module.base_kernel.lengthscale.detach().numpy()
                report_dict['lengthscale'] = '{:.2f}'.format(v[0][0])
        pbar.set_postfix(report_dict)
    
    def fit(self, X, y, y_var=None, n_iter=100, learning_rate=0.1):
        x = self.seq_to_one_hot(X)
        y = self.get_tensor(y)
        self.set_likelihood(y_var=y_var)
        self.model = self.to_device(GPModel(x, y, self.kernel, self.likelihood,
                                            output_device=self.output_device,
                                            n_devices=self.n_devices,
                                            train_mean=self.train_mean)) 

        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        pbar = tqdm(range(n_iter), desc='Iterations')
        for _ in pbar:
            optimizer.zero_grad()
            output = self.model(x)
            loss = -mll(output, y)
            loss.backward()
            optimizer.step()
            
            self.report_progress(pbar, loss)
    
    def predict(self, X):
        x = self.seq_to_one_hot(X)
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f_preds = self.model(x)
        # TODO: error when asking for variance: f_preds.variance
        return(f_preds.mean)
        