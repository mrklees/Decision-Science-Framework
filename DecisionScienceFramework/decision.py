# Core Python
from collections import OrderedDict
import pickle

# Anaconda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm
import patsy

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
        self.last_run['isLoss'] = np.where(self.last_run.loss < 0, 1, 0)

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
        loss_hist["isLoss"] = np.where(loss_hist.x < 0, 1, 0)
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

    def summary(self):
        """Method for displaying the current state of the decision
        
        We would ideally see all currently created variables. 
        """
        pass

    def evpi(self):
        for column in school_decision.random.keys():
            self.information_values[column] = self._calculate_evpi(column)
            print(f"{column}: {self._calculate_evpi(column)}")

    def _calculate_evpi(self, column):
        """Calculate information value for a single specified variable
        
        Our information value 
        """

        baseline_uncertainty = self.last_run['loss'].mean()
        mu = self.last_run[column].mean()
        fixed = self.last_run.copy()
        fixed[column] = mu
        for key, item in school_decision.deterministic.items():
            var = patsy.dmatrix(f"I({item}) - 1", data=fixed)
            fixed[key] = np.asarray(var)
        updated_uncertainty = fixed['loss'].mean()
        return updated_uncertainty - baseline_uncertainty
