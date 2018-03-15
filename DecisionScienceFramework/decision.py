from math import exp
import pickle

import numpy as np
from scipy.stats import lognorm, beta, norm
from scipy.optimize import fmin

import pymc3 as pm

import patsy


class Decision(object):
    def __init__(self, name, fp):
        self.name = name
        self.file_path = fp
        self.initialize_model()
        self.loss = None
        self.distribution_dict = {
            "Normal": pm.Normal,
            "Lognormal": pm.Lognormal,
            "Beta": pm.Beta,
            "Uniform": pm.Uniform,
            "Bernoulli": pm.Bernoulli,
            "Binomial": pm.Binomial
        }
        self.default_params_dict = {
            "Normal": {"mu": 0, "sd": 1},
            "Lognormal": {"mu": 0, "sd": 1},
            "Beta": {"alpha": 1, "beta": 1},
            "Uniform": {'lower': 0, 'upper': 1},
            "Bernoulli": {"p": 0.5},
            "Binomial": {"n": 1, "p": 0.5}
        }

    def add_variable(self, name, dist="Normal", params={"mu": 0, "sd": 1}):
        """Add a variable to the decision.

        This method allows us to build a decision model one variable
        at a time.  This is typically not how models are built in
        pymc3.
        Args:
            name (str): Label for the variable.
            dist (PyMC3 RV): PyMC3 Random Variable
            params (tuple): tuple containing required parameters
        """
        distribution = self.distribution_dict[dist]
        with self.model:
            vars()[name] = distribution(name=name, **params)

    def set_loss(self, formula_like):
        """Set loss using a patsy formula.

        See the documentation:
        https://patsy.readthedocs.io/en/latest/formulas.html
        for more information on patsy formulas. Currently we only support
        formulas that contain single word variables or you can wrap a
        multi-word variable in Q() like so: Q("Variable Name").
        """
        loss = patsy.dmatrix(f"I({formula_like}) - 1", data=self.last_run)
        self.last_run['loss'] = np.asarray(loss)

    def sample(self, nsamples):
        with self.model:
            trace = pm.sample(nsamples, tune=500)
            self.last_run = pm.backends.tracetab.trace_to_dataframe(trace)

    def save_state(self):
        fp = '\\'.join([self.file_path, f'{self.name}.pkl'])
        pickle.dump(obj=self, file=open(fp, 'wb'))

    def initialize_model(self):
        with pm.Model() as self.model:
            pass


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
                                       np.norm.rvs(size=2),
                                       ftol=1e-8)
        else:
            self.alpha, self.beta = fmin(self.beta_loss, (0, 1), ftol=1e-8)

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
