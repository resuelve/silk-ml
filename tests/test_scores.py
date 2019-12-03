import unittest

from silk_ml.general.scores import ls_score


class TestScores(unittest.TestCase):

    def test_ls_score(self):
        y = [0, 1]
        self.assertEqual(ls_score(y, [1, 0]), -1)
        self.assertEqual(ls_score(y, [0, 0]), -1)
        self.assertEqual(ls_score(y, [1, 1]), 0)
        self.assertEqual(ls_score(y, [0, 1]), 1)

        print(ls_score([1, 1, 1, 0, 1], [0, 0, 1, 0, 1]))


if __name__ == '__main__':
    unittest.main()
