import numpy as np
import torch
import gpytorch

from gpytorch.kernels.multi_device_kernel import MultiDeviceKernel
from tqdm import tqdm


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
    def __init__(self, kernel, likelihood, output_device=None, n_devices=None):
        self.kernel = kernel
        self.likelihood = likelihood
        self.output_device = output_device
        self.n_devices = n_devices
    
    def seq_to_one_hot(self, X):
        m = np.array([[a for a in x] for x in X])
        onehot = []
        for i in range(m.shape[1]):
            c = m[:, i]
            alleles = np.unique(c)
            for allele in alleles:
                onehot.append(torch.tensor(c == allele, dtype=torch.float32))
        onehot = torch.stack(onehot, 1)
        return(onehot)
    
    def fit(self, X, y, n_iter=100):
        x = self.seq_to_one_hot(X)
        model = GPModel(x, y, self.kernel, self.likelihood,
                        output_device=self.output_device,
                        n_devices=self.n_devices)
        if self.output_device:
            model = model.to(self.output_device)
        self.model = model 
        self.model.train()
        self.likelihood.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        for _ in tqdm(range(n_iter)):
            optimizer.zero_grad()
            output = self.model(x)
            loss = -mll(output, y)
            loss.backward()
            optimizer.step()
    
    def predict(self, X):
        x = self.seq_to_one_hot(X)
        f_preds = self.model(x)
        return(f_preds.mean, f_preds.variance)
        