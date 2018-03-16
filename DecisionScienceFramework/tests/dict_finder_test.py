from unittest import TestCase

from DecisionScienceFramework.utils import DistFinder
from scipy.stats import lognorm, beta


class TestDecision(TestCase):
    def test_lognormal_finder(self):
        DistFinder(0, 10, 4, lognorm)
        