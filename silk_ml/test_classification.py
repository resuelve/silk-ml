import unittest
from .classification import Classificator

class TestClassification(unittest.TestCase):

    def test_constructor(self):
        self.assertEqual(Classificator().target, None)
        self.assertEqual(Classificator('test').target, 'test')
        self.assertEqual(Classificator(target='test').target, 'test')
        self.assertEqual(Classificator('test', 'hello').target, 'test')


if __name__ == '__main__':
    unittest.main()
