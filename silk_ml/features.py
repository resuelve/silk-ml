import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind

from .plots import plot_categorical, plot_numerical


def split_classes(X, Y, label):
    ''' Returns the splited value of the dataset using the requested label

    :param X: Main dataset with the variables
    :type X: pd.DataFrame
    :param data: Main dataset
    :type data: pd.DataFrame
    :param target: Categorical variable to classify
    :type target: str or None
    :param label: Name of the variable to split
    :type label: str
    :return: The `positive` and `negative` data splited
    :rtype: tuple(pd.Series, pd.Series)
    '''
    positive = X.loc[Y == 1][label]
    negative = X.loc[Y != 1][label]
    return positive, negative


def features_metrics(X, Y, targetname, plot=None):
    ''' Determines the likelihood from each variable of splitting correctly the dataset

    :param X: Main dataset with the variables
    :type X: pd.DataFrame
    :param Y: Target variable
    :type Y: pd.Series
    :param targetname: Target name for reports
    :type targetname: str or None
    :param plot: Plots the variables, showing the difference in the classes
    :type plot: 'all' or 'categorical' or 'numerical' or None
    :return: Table of variables and their classification tests
    :rtype: pd.DataFrame
    '''
    plot_cat = plot in ['all', 'categorical']
    plot_num = plot in ['all', 'numerical']

    features = {}
    columns = X.columns.tolist()

    def is_categorical(column): return len(X[column].unique().tolist()) <= 2

    def test_variable(column):
        if is_categorical(column):
            test, plot = _test_categorical, plot_cat 
        else:
            test, plot = _test_numerical, plot_num 
        return test(X, Y, column, targetname, plot)
    
    features = {
        'cardinality kind': [
            'categorical' if is_categorical(column) else 'numerical'
            for column in columns
        ],
        'split probability': [
            f'{(100 - test_variable(column) * 100):.4f} %'
            for column in columns
        ],
    }
    return pd.DataFrame(features, index=columns)


def _test_categorical(X, Y, column, targetname, plot_cat):
    if plot_cat:
        plot_categorical(X, Y, column, targetname)
    cont_table = pd.crosstab(Y, X[column], margins=False)
    test = chi2_contingency(cont_table.values)
    return test[1]


def _test_numerical(X, Y, column, targetname, plot_num):
    positive, negative = split_classes(X, Y, column)
    if plot_num:
        plot_numerical(positive, negative, column, targetname)
    _, p_value = ttest_ind(positive, negative)
    return p_value
