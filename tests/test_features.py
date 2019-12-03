import unittest
import pandas as pd
import random as rd

from silk_ml.general.features import split_classes


class TestFeatures(unittest.TestCase):

    def test_split(self):
        x = {
            'label1': [rd.random() + 5 for _ in range(100)],
            'label2': [rd.random() * 3 - 1 for _ in range(100)],
            'label3': [round(rd.random()) for _ in range(100)],
        }
        X = pd.DataFrame(x)
        Y = pd.Series([round(rd.random()) for _ in range(100)])
        pos1, neg1 = split_classes(X, Y, 'label1')
        pos2, neg2 = split_classes(X, Y, 'label2')

        # Check that the sum of the values are equivalent to the amount
        self.assertEqual(len(pos1) + len(neg1), 100)
        self.assertEqual(len(pos2) + len(neg2), 100)


if __name__ == '__main__':
    unittest.main()
