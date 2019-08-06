import unittest
import random as rd
import pandas as pd
from sklearn import preprocessing
from silk_ml.classification import Classifier


def generate_test():
    '''Generates a dataframe with 100 random elements with:
        * label1: numerical variable, with median in 5 and std of 1 
        * label2: numerical variable, with median in -1 and std of 1.5
        * label3: categorical variable (0 or 1)
        * target: categorical variable (0 or 1)
    '''
    rd.seed(50)
    classifier = Classifier()
    data = {
        'label1': [rd.random() + 5 for _ in range(100)],
        'label2': [rd.random() * 3 - 1 for _ in range(100)],
        'label3': [round(rd.random()) for _ in range(100)],
        'target': [round(rd.random()) for _ in range(100)],
    }
    classifier.data = pd.DataFrame(data)
    classifier.set_target('target')
    return classifier


class TestClassification(unittest.TestCase):

    def test_constructor(self):
        self.assertEqual(Classifier().target, None)
        self.assertEqual(Classifier('test').target, 'test')
        self.assertEqual(Classifier(target='test').target, 'test')

    def test_standarize(self):
        Classifier = generate_test()

        # Applies a power normalizer and a min-max scaler
        normalizer = preprocessing.PowerTransformer()
        scaler = preprocessing.MinMaxScaler()
        Classifier.standarize(normalizer, scaler)

        # Check if all the vales are in the 0-1 range
        lower = list((Classifier.X < 0).any().to_dict().values())
        upper = list((Classifier.X > 1).any().to_dict().values())
        self.assertEqual(lower, [False, False, False])
        self.assertEqual(upper, [False, False, False])

    def test_split(self):
        Classifier = generate_test()
        pos1, neg1 = Classifier.split_classes('label1')
        pos2, neg2 = Classifier.split_classes('label2')

        # Check that the sum of the values are equivalent to the amount
        self.assertEqual(len(pos1) + len(neg1), 100)
        self.assertEqual(len(pos2) + len(neg2), 100)

    def test_metrics(self):
        Classifier = generate_test()
        metrics = Classifier.features_metrics()

        # Check kind of values, p-values are variables
        self.assertEqual(
            metrics.loc['cardinality kind'].tolist(),
            ['numerical', 'numerical', 'categorical']
        )


if __name__ == '__main__':
    unittest.main()
