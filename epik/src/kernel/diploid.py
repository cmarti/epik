import torch as torch

from torch.nn import Parameter

from epik.src.kernel.base import SequenceKernel


class DiploidKernel(SequenceKernel):
    is_stationary = True
    def __init__(self, **kwargs):
        super().__init__(n_alleles=2, seq_length=0, **kwargs)
        self.define_kernel_params()
    
    def define_kernel_params(self):
#         constraints = {'raw_log_lda': LessThan(upper_bound=0.),
#                        'raw_log_eta': LessThan(upper_bound=0.)}
        params = {'raw_log_lda': Parameter(torch.zeros(1)),
                  'raw_log_eta': Parameter(torch.zeros(1)),
                  'raw_log_mu': Parameter(torch.zeros(1))}
        self.register_params(params=params) #constraints=constraints
    
    @property
    def mu(self):
        return(torch.exp(self.raw_log_mu))
    
    @property
    def lda(self):
#         return(torch.exp(self.raw_log_lda_constraint.transform(self.raw_log_lda)))
        return(torch.exp(self.raw_log_lda))

    @property
    def eta(self):
#         return(torch.exp(self.raw_log_eta_constraint.transform(self.raw_log_eta)))
        return(torch.exp(self.raw_log_eta))

    def dist(self, x1, x2):
        return(x1.matmul(x2.transpose(-2, -1)))
    
    def calc_distance_classes(self, x1, x2):
        l = x1.shape[1]
        s1 = self.dist(x1[:, :, 1], x2[:, :, 1])
        s2 = self.dist(x1[:, :, 0], x2[:, :, 0]) + self.dist(x1[:, :, 2], x2[:, :, 2])
        d2 = self.dist(x1[:, :, 0], x2[:, :, 2]) + self.dist(x1[:, :, 2], x2[:, :, 0])
        d1 = l - s1 - s2 - d2
        return(s2, d2, s1, d1)
    
    def _forward(self, x1, x2, mu, lda, eta):
        s2, d2, s1, d1 = self.calc_distance_classes(x1, x2)
        kernel = ((mu + 2 * lda + eta)**s2) * ((mu - 2 * lda + eta)**d2) * ((mu + eta)**s1) * ((mu - eta)**d1)
        return(kernel)

    def forward(self, x1, x2, **params):
        kernel = self._forward(x1, x2, self.mu, self.lda, self.eta)
        return(kernel)


class GeneralizedDiploidKernel(DiploidKernel):
    def __init__(self, seq_length, **kwargs):
        super().__init__(seq_length=seq_length, **kwargs)
        self.define_kernel_params()
    
    def define_kernel_params(self):
        super().define_kernel_params()
        params = {'raw_logit_p': Parameter(torch.zeros(1))}
        self.register_params(params=params)
    
    @property
    def odds(self):
        return(torch.exp(self.raw_logit_p))
    
    def _forward(self, x1, x2, mu, lda, eta, odds):
        s2, d2, s1, d1 = self.calc_distance_classes(x1, x2)
        kernel = (mu + (1 + odds) * lda + odds * eta)**s2 * (mu - (1 + odds) * lda + odds * eta)**d2 * (mu + odds * eta)**s1 * (mu - eta)**d1
        return(kernel)

    def forward(self, x1, x2, **params):
        kernel = self._forward(x1, x2, self.mu, self.lda, self.eta, self.odds)
        return(kernel)
