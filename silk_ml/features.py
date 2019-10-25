import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind

from .plots import plot_categorical, plot_numerical


def split_classes(X, Y, label):
    """ Returns the splited value of the dataset using the requested label

    Args:
        X (pd.DataFrame): Main dataset with the variables
        Y (pd.Series): Target variable
        label (str): Name of the variable to split

    Returns:
        tuple(pd.Series, pd.Series): The `positive` and `negative` data splited
    """
    positive = X.loc[Y == 1][label]
    negative = X.loc[Y != 1][label]
    return positive, negative


def features_metrics(X, Y, target_name, plot=None):
    """ Determines the likelihood from each variable of splitting correctly the dataset

    Args:
        X (pd.DataFrame): Main dataset with the variables
        Y (pd.Series): Target variable
        target_name (str or None): Target name for reports
        plot ('all' or 'categorical' or 'numerical' or None): Plots the
            variables, showing the difference in the classes

    Returns:
        pd.DataFrame: Table of variables and their classification tests
    """
    plot_cat = plot in ['all', 'categorical']
    plot_num = plot in ['all', 'numerical']

    features = {}
    columns = X.columns.tolist()

    def is_categorical(column):
        # Currify the categorical validation
        return len(X[column].unique().tolist()) <= 2

    def test_variable(column):
        # Currify the call for the p-value calculator
        if is_categorical(column):
            test, plot = _test_categorical, plot_cat
        else:
            test, plot = _test_numerical, plot_num
        return test(X, Y, column, target_name, plot)

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


def _test_categorical(X, Y, column, target_name, plot_cat):
    """ Runs the p-value test for the current variable
    
    Args:
        X (pd.DataFrame): Main dataset with the variables
        Y (pd.Series): Target variable
        column (str): Name of the variable to test
        target_name (str or None): Target name for reports
        plot_cat (bool): Plots the current variable
    
    Returns:
        float: p-value of the variables
    """
    if plot_cat:
        plot_categorical(X, Y, column, target_name)
    cont_table = pd.crosstab(Y, X[column], margins=False)
    test = chi2_contingency(cont_table.values)
    return test[1]


def _test_numerical(X, Y, column, target_name, plot_num):
    """ Runs the p-value test for the current variable
    
    Args:
        X (pd.DataFrame): Main dataset with the variables
        Y (pd.Series): Target variable
        column (str): Name of the variable to test
        target_name (str or None): Target name for reports
        plot_num (bool): Plots the current variable
    
    Returns:
        float: p-value of the variables
    """
    positive, negative = split_classes(X, Y, column)
    if plot_num:
        plot_numerical(positive, negative, column, target_name)
    _, p_value = ttest_ind(positive, negative)
    return p_value
