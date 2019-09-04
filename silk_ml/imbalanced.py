import pandas as pd

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours


def resample(X, Y, rate=0.9, strategy='hybrid'):
    ''' Sampling based methods to balance dataset

    :param X: Main dataset with the variables
    :type X: pd.DataFrame
    :param Y: Target variable
    :type Y: pd.Series
    :param rate: Ratio of the number of samples in the minority class over the
        number of samples in the majority class after resampling
    :type rate: float
    :param strategy: Strategy to balance the dataset
    :type strategy: 'hybrid' | 'over_sampling' | 'under_sampling'
    '''
    strategies = {
        'hybrid': SMOTEENN(sampling_strategy=rate),
        'over_sampling': SMOTE(sampling_strategy=rate),
        'under_sampling': EditedNearestNeighbours(),
    }
    resampling = strategies[strategy]
    cols = X.columns
    X_r, Y_r = resampling.fit_resample(X, Y)
    return pd.DataFrame(data=X_r, columns=cols), Y_r
