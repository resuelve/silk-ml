import random as rd
import pandas as pd

from silk_ml.classification import Classifier


def generate_test():
    ''' Generates a dataframe with 100 random elements with:
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
    classifier.target = 'target'
    return classifier
