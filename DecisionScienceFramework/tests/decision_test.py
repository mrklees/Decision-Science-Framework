from unittest import TestCase

from DecisionScienceFramework.decision import Decision


class TestDecision(TestCase):
    def setUp(self):
        self.model = Decision('model', fp='C:/Users/perus/Desktop')            

    def test_add_vars(self):       
        self.model.add_variable('benefits')
        self.model.add_variable('costs')
        self.assertEqual(len(self.model.model.free_RVs), 2)
        
    def test_sample(self):
        self.model.add_variable('benefits')
        self.model.add_variable('costs')
        self.model.sample(1000)
        self.assertIsNotNone(self.model.last_run)

    def test_loss(self):
        self.model.add_variable('benefits')
        self.model.add_variable('costs')
        self.model.sample(1000)
        self.model.set_loss('benefits - costs')
        self.assertTrue('loss' in self.model.last_run.columns)
