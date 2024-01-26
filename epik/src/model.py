import torch
import gpytorch

from time import time
from tqdm import tqdm

from gpytorch.means import ZeroMean, ConstantMean
from gpytorch.kernels.multi_device_kernel import MultiDeviceKernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.likelihoods.gaussian_likelihood import (FixedNoiseGaussianLikelihood,
                                                      GaussianLikelihood)
from epik.src.LBFGS import FullBatchLBFGS
from epik.src.utils import get_tensor, to_device, split_training_test
 


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, kernel, likelihood,
                 device=None, n_devices=None, train_mean=False):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        
        if train_mean:
            self.mean_module = ConstantMean()
        else:
            self.mean_module = ZeroMean()
        
        self.covar_module = kernel
        # if device is None:
        #     self.covar_module = kernel
        # else:
        #     self.covar_module = MultiDeviceKernel(kernel,
        #                                           device_ids=range(n_devices),
        #                                           output_device=device)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class EpiK(object):
    def __init__(self, kernel, likelihood_type='Gaussian',
                 device=None, n_devices=1,
                 train_mean=False, train_noise=False,
                 learning_rate=0.1,
                 preconditioner_size=0, dtype=torch.float32,
                 track_progress=False, 
                 optimizer='Adam'):
        self.kernel = kernel
        self.likelihood_type = likelihood_type
        self.device = device
        self.train_mean = train_mean
        self.learning_rate = learning_rate
        self.n_devices = n_devices
        self.dtype = dtype
        self.preconditioner_size = preconditioner_size
        self.train_noise = train_noise
        self.track_progress = track_progress
        self.fit_time = 0
        self.optimizer_label = optimizer
    
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

        if self.device is not None:
            likelihood = likelihood.cuda()
        self.likelihood = likelihood # self.to_device(likelihood)
    
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
        return(get_tensor(ndarray, dtype=self.dtype, device=self.device))
    
    def set_training_mode(self):
        self.model.train()
        self.likelihood.train()
        
    def set_evaluation_mode(self):
        self.model.eval()
        self.likelihood.eval()
    
    def set_preconditioner_size(self, preconditioner_size=None):
        if preconditioner_size is None:
            preconditioner_size = self.preconditioner_size
        return(gpytorch.settings.max_preconditioner_size(preconditioner_size))
    
    def define_optimizer(self):
        if self.optimizer_label == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.fit = self.adam_fit
        elif self.optimizer_label == 'LBFGS':
            self.optimizer = FullBatchLBFGS(self.model.parameters(), lr=self.learning_rate)
            self.fit = self.lbfgs_fit
        else:
            msg = 'Optimizer {} not allowed'.format(self.optimizer_label)
            raise ValueError(msg)
    
    def define_loss(self):
        self.calc_mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
    
    def define_model(self):
        self.model = GPModel(self.X, self.y, self.kernel, self.likelihood,
                             train_mean=self.train_mean,
                             device=self.device,
                             n_devices=self.n_devices) # self.to_device()
        if self.device is not None:
            self.model = self.model.cuda()
    
    def get_gp_mean(self):
        if hasattr(self.model.mean_module, 'constant'):
            return(self.to_numpy(self.model.mean_module.constant))
        else:
            return(torch.zeros(1))
    
    def set_data(self, X, y, y_var=None):
        self.X = self.get_tensor(X)
        self.y = self.get_tensor(y)
        self.y_var = y_var
        
        self.set_likelihood(y_var=self.y_var)
        self.define_model()
        self.define_optimizer()
        self.define_loss()
    
    def adam_training_step(self, X, y):
        self.optimizer.zero_grad()
        self.loss = -self.calc_mll(self.model(X), y)
        self.loss.backward()
        self.optimizer.step()
        
        if hasattr(self.kernel, 'get_params'):
            self.params_history.append(self.get_params())
            self.loss_history.append(self.loss.detach().item())
    
    def adam_fit(self, n_iter=100):
        self.set_training_mode()
        
        pbar = range(n_iter)
        self.loss_history = []
        self.params_history = []
        
        with self.set_preconditioner_size():
            
            if n_iter > 1 and self.track_progress:
                pbar = tqdm(pbar, desc='Maximizing Marginal Likelihood')
            
            t0 = time()
            
            for _ in pbar:
                # # x = self.kernel.logit_rho.detach().cpu().numpy()
                # x = self.kernel.base_kernel.lengthscale.detach().cpu().numpy()[0]
                # x = sorted(x.flatten())
                # # print(x[:2], x[-2:], self.kernel.outputscale.detach().cpu().numpy())
                # print(x, self.kernel.outputscale.detach().cpu().numpy())
                self.adam_training_step(self.X, self.y)
                if n_iter > 1:
                    self.report_progress(pbar)
                    
            self.fit_time = time() - t0
    
    def lbfgs_fit(self, n_iter=100):
        
        self.loss_history = []
        self.params_history = []
        
        def closure():
            self.optimizer.zero_grad()
            loss =  -self.calc_mll(self.model(self.X), self.y)
            return(loss)

        loss = closure()
        loss.backward()

        with self.set_preconditioner_size():
            pbar = range(n_iter)
            
            if n_iter > 1 and self.track_progress:
                pbar = tqdm(pbar, desc='Maximizing Marginal Likelihood')
                
            t0 = time()
            for i in pbar:
                options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
                self.loss, _, _, _, _, _, _, fail = self.optimizer.step(options)
                
                if n_iter > 1:
                    self.report_progress(pbar)
        
                if hasattr(self.kernel, 'get_params'):
                    self.params_history.append(self.get_params())
                    self.loss_history.append(self.loss.detach().item())
        
                if fail:
                    print('Convergence reached at iteration {}!'.format(i))
                    break
            self.fit_time = time() - t0
    
    def predict(self, pred_X, calc_variance=False):
        if calc_variance:
            msg = 'Variance calculation not implemented yet'
            raise ValueError(msg)
        
        t0 = time()
        
        self.set_evaluation_mode()
        pred_X = self.get_tensor(pred_X)
        
        with torch.no_grad(), self.set_preconditioner_size(), gpytorch.settings.skip_posterior_variances(): #, , gpytorch.settings.fast_pred_var():
            f_preds = self.model(pred_X).mean

        self.pred_time = time() - t0
        return(f_preds)
    
    def get_prior(self, X, sigma2):
        self.X = X
        likelihood = FixedNoiseGaussianLikelihood(noise=sigma2 * torch.ones(X.shape[0]))
        model = self.to_device(GPModel(None, None, self.kernel, likelihood,
                                       train_mean=self.train_mean,
                                       device=self.device,
                                       n_devices=self.n_devices))
        prior = model.forward(X)
        return(prior)
    
    def sample(self, X, n=1, sigma2=1e-4):
        prior = self.get_prior(X, sigma2=sigma2)
        v = torch.zeros(n)
        with torch.no_grad(), self.set_preconditioner_size():
            y = prior.rsample(v.size())
        return(y)
    
    def get_params(self):
        params = {'mean': self.to_numpy(self.get_gp_mean())}
        if hasattr(self.kernel, 'get_params'):
            for param, values in self.kernel.get_params().items():
                params[param] = self.to_numpy(values)
        return(params)
    
    def simulate_dataset(self, X, sigma=0, ptrain=0.8):
        y_true = self.sample(X, n=1).flatten()
        y_true = y_true / y_true.std()
        
        splits = split_training_test(X, y_true, y_var=None, ptrain=ptrain)
        train_x, train_y, test_x, test_y, train_y_var = splits
        if sigma > 0:
            train_y = torch.normal(train_y, sigma)
            train_y_var = torch.full_like(train_y, sigma**2)
            
        return(train_x, train_y, test_x, test_y, train_y_var)
    
    def save(self, fpath):
        torch.save(self.model.state_dict(), fpath)
        
    def load(self, fpath):
        self.model.load_state_dict(torch.load(fpath))
