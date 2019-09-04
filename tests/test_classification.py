import unittest

from sklearn import preprocessing
from silk_ml.classification import Classifier
from .helper import generate_test


class TestClassification(unittest.TestCase):

    def test_constructor(self):
        self.assertEqual(Classifier().target, None)
        self.assertEqual(Classifier('test').target, 'test')
        self.assertEqual(Classifier(target='test').target, 'test')

    def test_metrics(self):
        Classifier = generate_test()
        metrics = Classifier.features_metrics()

        # Check kind of values, p-values are variables
        self.assertEqual(
            metrics['cardinality kind'].tolist(),
            ['numerical', 'numerical', 'categorical']
        )


if __name__ == '__main__':
    unittest.main()
