import numpy as np
import gc
import torch
import gpytorch

from time import time
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
                 partition_size=0, learning_rate=0.1,
                 preconditioner_size=0, dtype=torch.float32):
        self.kernel = kernel
        self.likelihood_type = likelihood_type
        self.output_device = output_device
        self.learning_rate = learning_rate
        self.n_devices = n_devices
        self.dtype = dtype
        self.partition_size = partition_size
        self.preconditioner_size = preconditioner_size
    
    def optimize_partition_size(self):
        # TODO: does not seem to be working
        '''
        Function to set up GPU settings like number of GPUs and the 
        partition size. Adapted from gpytorch website:
        
        https://docs.gpytorch.ai/en/stable/examples/02_Scalable_Exact_GPs/Simple_MultiGPU_GP_Regression.html
        
        '''
        
        N = self.X.size(0)
        partition_sizes = np.append(0, np.ceil(N / 2 ** np.arange(1, np.floor(np.log2(N)))).astype(int))
        for partition_size in partition_sizes:
            try:
                self.partition_size = partition_size
                self.fit(n_iter=3)
                break
            
            except (RuntimeError, AttributeError) as error:
                print(error)
                # handle CUDA OOM error
                gc.collect()
                torch.cuda.empty_cache()
    
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
    
    def report_progress(self, pbar, rho=None):
        loss = self.loss
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
    
    def training_step(self, X, y):
        self.optimizer.zero_grad()
        self.loss = -self.calc_mll(self.model(X), y)
        self.loss.backward()
        self.optimizer.step()
        self.loss_history.append(self.loss.detach().item())
        
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
                                            output_device=self.output_device,
                                            n_devices=self.n_devices))
    
    def get_gp_mean(self):
        c = self.model.mean_module.constant
        if self.output_device is not None:
            c = c.cpu()
        return(c.detach().numpy()) 
    
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
            
            if n_iter > 1:
                pbar = tqdm(pbar, desc='Maximizing Marginal Likelihood')
            
            t0 = time()    
            for _ in pbar:
                self.training_step(self.X, self.y)
                if n_iter > 1:
                    self.report_progress(pbar)
                    
            self.fit_time = time() - t0
    
    def predict(self, pred_X):
        self.set_evaluation_mode()
        
        t0 = time()
        with torch.no_grad(), self.set_preconditioner_size(), self.set_partition_size(): #, , gpytorch.settings.fast_pred_var():
            f_preds = self.model(self.get_tensor(pred_X)).mean
        # TODO: error when asking for variance: f_preds.variance
        self.pred_time = time() - t0
        return(f_preds)
