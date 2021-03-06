# Core Python
from collections import OrderedDict
from itertools import product
from math import ceil
import pickle

# Anaconda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import patsy
import pymc3 as pm

from DecisionScienceFramework.utils import DistFinder


class Decision(object):
    def __init__(self, name, fp):
        self.name = name
        self.file_path = fp
        self.initialize_model()
        self.last_run = None
        self.random = OrderedDict()
        self.deterministic = OrderedDict()
        self.information_values = OrderedDict()
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

    def add_variable_from_ci(self, name, dist, lower, upper, mode=None):
        # Calculate parameters 
        params = DistFinder(lower=lower, upper=upper, ev=mode, dist=dist).optimize()
        self.add_variable_from_params(name, dist, params=params)

    def add_variable_from_params(self, name, dist="Normal", params={"mu": 0, "sd": 1}):
        """Add a variable to the decision.

        This method allows us to build a decision model one variable
        at a time.  This is typically not how models are built in
        pymc3.
        Args:
            name (str): Label for the variable.
            dist (PyMC3 RV): A string describing the desired distribution
            params (tuple): tuple containing required parameters
        """
        # What if they submit a dist that doesn't exist?
        try:
            distribution = self.distribution_dict[dist]
        except KeyError:
            print("Input distrubtion not supported.  Choose one of the" +
                  f" following: {self.distribution_dict.keys()}")
            return None
        # What if they don't submit all of the required parameters?
        # TODO
        self.random[name] = params
        with self.model:
                vars()[name] = distribution(name=name, **params)

    def add_variable_from_patsy(self, name, formula_like):
        """We are often constructing new deterministic vars from existing ones
        
        Requires that samples have already been drawn.
        See the documentation:
        https://patsy.readthedocs.io/en/latest/formulas.html
        for more information on patsy formulas. Currently we only support
        formulas that contain single word variables or you can wrap a
        multi-word variable in Q() like so: Q("Variable Name").
        """
        if self.last_run is None:
            self.sample(1000)
        self.deterministic[name] = formula_like
        var = patsy.dmatrix(f"I({formula_like}) - 1", data=self.last_run)
        self.last_run[name] = np.asarray(var)

    def set_loss(self, formula_like):
        """Set loss using a patsy formula.

        We split this out only so that we could norm on the loss column name.
        """
        self.add_variable_from_patsy('loss', formula_like)
        self.last_run['isLoss'] = np.where(self.last_run.loss > 0, 1, 0)

    def sample(self, nsamples):
        with self.model:
            # TODO: Figure out what the fucking deal is with running njobs > 1
            # Ends with a compile error more times than not.
            trace = pm.sample(nsamples, tune=500, njobs=1)
            self.last_run = pm.backends.tracetab.trace_to_dataframe(trace)

    def save_state(self):
        fp = '\\'.join([self.file_path, f'{self.name}.pkl'])
        pickle.dump(obj=self, file=open(fp, 'wb'))

    def initialize_model(self):
        with pm.Model() as self.model:
            pass

    def plot_loss(self):
        """Plot a histogram of the losses"""
        # Calculate a histogram in numpy 
        hist, edges = np.histogram(self.last_run.loss, bins=50, density=True)
        # Fix edges so that they are midpoints of the different slices instead
        # of defining the edges of each slice
        midpoints = np.mean([edges[:-1].reshape(-1, 1), 
                             edges[1:].reshape(-1, 1)], axis=0)
        # Fit in dataframe and add flag for points < 0
        loss_hist = pd.DataFrame(np.hstack([midpoints.astype('int'), 
                                            hist.reshape(-1, 1)]), 
                                 columns=["x", "y"])
        loss_hist["isLoss"] = np.where(loss_hist.x > 0, 1, 0)
        # Do the plotting
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        sns.barplot("x", "y", hue="isLoss", data=loss_hist, ax=axs)
        # We tend to have a very tall spike.  Set the hight of this
        # distribution equal to the average of the 10 highest values
        hist.sort()
        axs.set_ylim(0, hist[-10:].mean())
        axs.legend(loc=1)
        plt.xticks(rotation=65)
        plt.show()

    def plot_variables(self):
        """Plot distributions of all input variables"""
        samples = self.last_run
        cols = samples.columns
        ncols = len(cols)
        if ncols <= 8:
            grid_cols = 2
            grid_rows = ceil(ncols/2)       
        else:
            grid_cols = 3
            grid_rows = ceil(ncols/3)
        
        fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(30, 30))
        for ix, coord in enumerate(product(range(0, grid_rows), range(0, grid_cols))):
            try:
                axs[coord].hist(samples[cols[ix]], bins=50)
                axs[coord].set_xlabel(cols[ix])
                #axs[coord].rc("axes", titlesize=20)
            except:
                pass
        
        plt.show()

    def summary(self):
        """Method for displaying the current state of the decision
        
        We would ideally see all currently created variables. 
        """
        pass

    def evpi(self):
        for column in self.random.keys():
            self.information_values[column] = self._calculate_evpi(column)
            print(f"{column}: {self._calculate_evpi(column)}")

    def _calculate_evpi(self, column):
        """Calculate information value for a single specified variable
        
        Our information value 
        """

        baseline_uncertainty = self.eol(self.last_run)
        mu = self.last_run[column].mean()
        fixed = self.last_run.copy()
        fixed[column] = mu
        for key, item in self.deterministic.items():
            var = patsy.dmatrix(f"I({item}) - 1", data=fixed)
            fixed[key] = np.asarray(var)
        updated_uncertainty = self.eol(fixed)
        return baseline_uncertainty - updated_uncertainty

    @staticmethod
    def eol(samples):
        """Calculate the Expected Opportunity Loss for a given set of samples
        
        First we consider the Expected Loss over our sample. This indicates 
        what our "default" position should be. If it is negative (which 
        corresponds to a positive outcome), then we decide to accepting the
        proposal.  Otherwise if the expected loss is positive, then we default
        to declining the proposal.  Given that default, the Expected Opportunity
        Loss is thus the average outcome if we had done the opposite.  So we 
        sum up the losses where, if we accepted the proposal, then that 
        declining the proposal would have actually been the right decision and
        divide that over the total number of simulations considered.
        """
        exp_loss = samples.loss.mean()
        if exp_loss < 0:
            return sum(samples[samples.loss > 0].loss) / samples.shape[0]
        else:
            return sum(samples[samples.loss < 0].loss) / samples.shape[0]

    @staticmethod
    def get_interval(alpha, samples):
        """Yields the interval between alhpa and 1-alpha
        
        Use this for calculations like "95% of values fall between (a, b)"
        """
        loss_vec = samples.sort_values('loss', ascending=True).loss
        min_ix = int(loss_vec.shape[0]*alpha)
        max_ix = int(loss_vec.shape[0]*(1-alpha))
        return loss_vec.iloc[min_ix], loss_vec.iloc[max_ix]  
