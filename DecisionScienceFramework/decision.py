import pickle
import numpy as np
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
            dist (PyMC3 RV): A string describing the desired distribution
            params (tuple): tuple containing required parameters
        """
        # What if they submit a dist that doesn't exist?
        try:
            distribution = self.distribution_dict[dist]
        except KeyError:
            print(f"Input distrubtion not supported.  Choose one of the" +
                  f" following: {self.distribution_dict.keys()}")
            return None
        # What if they don't submit all of the required parameters?
        # TODO
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
