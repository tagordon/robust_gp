import numpy as np
import celerite2
from scipy.special import loggamma

class student_t_GP(celerite2.GaussianProcess):
    
    def __init_(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
    def log_likelihood(self, y, nu=1):
        
        d = len(y)    
        return (-0.5 * (nu + d) * np.log(1 + np.sum(y * self.apply_inverse(y)) / nu)
            - 0.5 * (self._log_det + d * np.log(nu * np.pi))
            + loggamma(0.5 * (nu + d)) - loggamma(0.5 * nu))
    
class biased_GP(celerite2.GaussianProcess):
    
    def __init__(self, *args, bias=0.0, **kwargs):
        
        self.b = bias
        super().__init__(*args, **kwargs)
        
    def log_likelihood(self, y, lam=0.0):
        
        return super().log_likelihood(y + self.b) - lam * np.sum(np.abs(self.b))
    
    def predict(self, y, **kwargs):
        
        return super().predict(y + self.b, **kwargs)
    
def aic(gp, y, return_bias=False):
    
    ky = gp.apply_inverse(y)
    a = np.zeros_like(y)
    b = np.zeros_like(y)
    
    for i in range(len(y)):
        z = np.zeros_like(y)
        z[i] = 1.0
        kz = gp.apply_inverse(z)
        dg = ky[i] + np.dot(y + z, kz)
        aii = kz[i]
    
        b[i] = 0.5 - dg / (2 * aii)
        a[i] = b[i] * (dg - aii + b[i] * aii)
    
    if return_bias:
        return 2 * (a + 1), b
    else:
        return 2 * (a + 1)
    
def bic(gp, y, return_bias=False):
    
    if return_bias:
        a, b = aic(gp, y, return_bias=True)
        return a - 2 + np.log(len(y)), b
    else:
        return aic(gp, y) - 2 + np.log(len(y))