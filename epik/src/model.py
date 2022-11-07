import numpy as np
import gc
import torch
import gpytorch

from tqdm import tqdm
from gpytorch.kernels.multi_device_kernel import MultiDeviceKernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.likelihoods.gaussian_likelihood import (FixedNoiseGaussianLikelihood,
                                                      GaussianLikelihood)

from epik.src.utils import get_tensor, to_device, get_gpu_memory


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, kernel, likelihood,
                 output_device=None, n_devices=None):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
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
                 dtype=torch.float32):
        self.kernel = kernel
        self.likelihood_type = likelihood_type
        self.output_device = output_device
        self.n_devices = n_devices
        self.dtype = dtype
        self.partition_size = None
        self.preconditioner_size = 100
    
    def optimize_partition_size(self, X, y, y_var=None, preconditioner_size=100):
        '''
        Function to set up GPU settings like number of GPUs and the 
        partition size. Adapted from gpytorch website:
        
        https://docs.gpytorch.ai/en/stable/examples/02_Scalable_Exact_GPs/Simple_MultiGPU_GP_Regression.html
        
        '''
        
        N = X.size(0)
        partition_sizes = np.append(0, np.ceil(N / 2 ** np.arange(1, np.floor(np.log2(N)))).astype(int))
        # partition_sizes = partition_sizes[1:]
        pbar = tqdm(partition_sizes, desc='Optimizing GPU settings')
    
        for partition_size in pbar:
            report = {'Number of devices': str(self.n_devices), 
                      'Kernel partition size': str(partition_size)}
            pbar.set_postfix(report)
            
            try:
                self.fit(X, y, y_var=y_var, n_iter=1,
                         partition_size=partition_size, 
                         preconditioner_size=preconditioner_size)
                break
            
            except (RuntimeError, AttributeError) as error:
                report['Status'] = error
                
            finally:
                # handle CUDA OOM error
                gc.collect()
                torch.cuda.empty_cache()
                
        self.partition_size = partition_size
        self.preconditioner_size = preconditioner_size
    
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
        report_dict = {'loss': '{:.3f}'.format(loss.detach().numpy()),
                       'mem': get_gpu_memory(self.output_device)}
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
    
    def _fit(self, X, y, loss_function, optimizer, n_iter):
        pbar = range(n_iter)
        
        if n_iter > 1:
            pbar = tqdm(pbar, desc='Iterations')
            
        for _ in pbar:
            try:
                optimizer.zero_grad()
                output = self.model(X)
                loss = -loss_function(output, y)
                loss.backward()
                optimizer.step()
                self.loss = loss
                
                if n_iter > 1:
                    self.report_progress(pbar, loss)
                    
            except KeyboardInterrupt:
                break
    
    def set_training_mode(self):
        self.model.train()
        self.likelihood.train()
        
    def set_evaluation_mode(self):
        self.model.eval()
        self.likelihood.eval()
    
    def fit(self, X, y, y_var=None, n_iter=100, learning_rate=0.1,
            partition_size=None, preconditioner_size=None):
        
        if partition_size is None:
            partition_size = self.partition_size
        
        if preconditioner_size is None:
            preconditioner_size = self.preconditioner_size
        
        X, y = self.get_tensor(X), self.get_tensor(y)

        self.set_likelihood(y_var=y_var)
        self.model = self.to_device(GPModel(X, y, self.kernel, self.likelihood,
                                            output_device=self.output_device,
                                            n_devices=self.n_devices)) 
        self.set_training_mode()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

        if partition_size is not None and preconditioner_size is not None:
            with gpytorch.beta_features.checkpoint_kernel(partition_size), \
             gpytorch.settings.max_preconditioner_size(preconditioner_size):
                self._fit(X, y, mll, optimizer, n_iter)
        else:
            self._fit(X, y, mll, optimizer, n_iter)
    
    def predict(self, pred_X, partition_size=None):
        self.set_evaluation_mode()
        
        if partition_size is None:
            partition_size = self.partition_size
        
        if partition_size is None:
            with torch.no_grad(): #, gpytorch.settings.fast_pred_var():
                f_preds = self.model(self.get_tensor(pred_X))
        else:
            with torch.no_grad(), gpytorch.beta_features.checkpoint_kernel(partition_size):
                f_preds = self.model(self.get_tensor(pred_X))
                
        # TODO: error when asking for variance: f_preds.variance
        return(f_preds.mean)
