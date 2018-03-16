import numpy as np
from scipy.stats import lognorm, beta, norm
from scipy.optimize import fmin
from math import exp


class DistFinder(object):
    """Find parameters for any confidence interval.

    There are two general cases.  First where, given a 90% CI, I can directly
    calculate the parameters of the distribution.  This is the symmetrical
    case.  Otherwise, the parameters have to be estimated based on a
    provdided estimate of the mean of the distribution.
    Class for approximating the paramteters of a log-normal distribution or
    a beta distribution.  Because they are flexible, asymetric distributions,
    we need to incorporate the
    TODO: Add support for distributions that are currently in actions.py
    and are symmetrical.
    """
    def __init__(self, lower, upper, ev, dist):
        self.lower_bound = lower
        self.upper_bound = upper
        self.ev = ev
        self.dist = dist
        self.optimize()
        self.summary()
        self.plot_fit()

    def lognormal_loss(self, x):
        mu, sigma = x
        prop_mean = lognorm.mean(sigma, scale=exp(mu))
        prop_low, prop_high = lognorm.interval(0.9, sigma, scale=exp(mu))
        return (prop_mean - self.ev)**2 + \
               (self.upper_bound - prop_high)**2 + \
               (self.lower_bound - prop_low)**2

    def beta_loss(self, x):
        a, b = x
        prop_mean = beta.mean(a, b)
        prop_low, prop_high = beta.interval(0.9, a, b)
        return (prop_mean - self.ev)**2 + \
               (self.upper_bound - prop_high)**2 + \
               (self.lower_bound - prop_low)**2

    def optimize(self):
        if self.dist == lognorm:
            self.mu, self.sigma = fmin(self.lognormal_loss,
                                       np.zeros(2),
                                       ftol=1e-8)
            return self.mu, self.sigma                           
        else:
            self.alpha, self.beta = fmin(self.beta_loss,
                                         (0.5, 0.5),
                                         ftol=1e-8)
            return self.alpha, self.beta

    def plot_fit(self):
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axs = plt.subplots()
        if self.dist == lognorm:
            sns.distplot(self.dist.rvs(self.sigma, scale=exp(self.mu), 
                                       size=1000), ax=axs)
        else:
            sns.distplot(self.dist.rvs(self.alpha, self.beta,
                                       size=1000), ax=axs)
    
    def summary(self):
        if self.dist == lognorm:
            final_mean = lognorm.mean(self.sigma, scale=exp(self.mu))
            final_interval = lognorm.interval(0.9, self.sigma, 
                                                scale=exp(self.mu))
            print(f"The fit parameters for this lognormal are mu: {self.mu}, s: {self.sigma} ")
        else:
            final_mean = beta.mean(a=self.alpha, b=self.beta)
            final_interval = beta.interval(0.9, a=self.alpha, b=self.beta)
            print(f"The fit parameters for this beta are alpha: {self.alpha}, beta: {self.beta}")
        
        print(f"The fit distribution has a mean of {final_mean} and a 90% CI" +
              f" of {final_interval}.")