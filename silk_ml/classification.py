import pandas as pd

from .features import features_metrics
from .plots import plot_corr, plot_mainfold, plot_roc_cross_val
from .imbalanced import resample


class Classifier:
    ''' General tasks for classification and data analysis

    :param target: Categorical variable to classify
    :type target: str or None
    :param filename: Name with path for reading a csv file
    :type filename: str or None
    :param targetname: Target name for reports
    :type targetname: str or None
    '''

    def __init__(self, target=None, filename=None, targetname=None):
        pd.set_option('display.max_columns', None)
        self._target = target
        self.targetname = targetname
        if (filename and target):
            self.read_csv(target, filename)

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, target):
        ''' Sets the target variable and if the data value exists,
        the X and Y values are setted as well

        :param target: Categorical variable to classify
        :type target: str
        :param targetname: Target name for reports
        :type targetname: str or None
        '''
        self._target = target
        if self.data is not None:
            self.Y = self.data[target]
            self.X = self.data.drop(columns=[target])

    def read_csv(self, target, filename):
        ''' Reads a CSV file and separate the X and Y variables

        :param target: Categorical variable to classify
        :type target: str
        :param filename: Name with path for reading a csv file
        :type filename: str
        :return: `X`, `Y`, and `data` values
        :rtype: list(pd.DataFrame)
        '''
        self.data = pd.read_csv(filename)
        self.target = target
        return self.X, self.Y, self.data

    def standardize(self, normalizer, scaler):
        ''' Applies a normalizer and scaler preprocessing steps

        :param normalizer: Class that centers the data
        :type normalizer: Class.fit_transform
        :param scaler: Class that modifies the data boundaries
        :type scaler: Class.fit_transform
        '''
        normalized = normalizer.fit_transform(self.X).transpose()

        # Check if in the normalization any data get lost
        for i, column in enumerate(self.X.columns.tolist()):
            if normalized[i].var() <= 1e-10:
                normalized[i] = self.X[column]

        return scaler.fit_transform(normalized.transpose())

    def features_metrics(self, plot=None):
        ''' Checks for each variable the probability of being splited

        :param plot: Plots the variables, showing the difference in the classes
        :type plot: 'all' or 'categorical' or 'numerical' or None
        :return: Table of variables and their classification tests
        :rtype: pd.DataFrame
        '''
        return features_metrics(self.X, self.Y, self.targetname, plot)

    def remove_features(self, features):
        ''' Remove features from the X values

        :param features: Column's names to remove
        :type features: list(str)
        '''
        self.X = self.X.drop(columns=features)

    def resample(self, rate=0.9, strategy='hybrid'):
        ''' Sampling based methods to balance dataset

        :param rate: Ratio of the number of samples in the minority class over
            the number of samples in the majority class after resampling
        :type rate: float
        :param strategy: Strategy to balance the dataset
        :type strategy: 'hybrid' or 'over_sampling' or 'under_sampling'
        '''
        self.X, self.Y = resample(self.X, self.Y, rate, strategy)

    def cross_validation(self, models, scores, folds=30, folds=):
        ''' Validates several models and scores

        :param models: Models to evaluate
        :type models: list(tuple)
        :param scores: Scores to measure the models
        :type scores: list(tuple)
        :param folds: Number of folds in a (Stratified)KFold
        :type folds: int
        '''
        return cross_validation(self.X, self.Y, models, scores, folds)

    def plot_corr(self, values=True):
        ''' Plots the correlation matrix

        :param values: Shows each of the correlation values
        :type values: bool
        '''
        plot_corr(self.data, values)

    def plot_mainfold(self, method):
        ''' Plots the reduced space using a mainfold transformation

        :param method: Mainfold transformation method
        :type method: Class.fit_transform
        '''
        plot_mainfold(method, self.data, self.targetname)

    def plot_roc_cross_val(self, models):
        ''' Plots all the models with their ROC

        :param models: Models to evaluate
        :type models: list(tuple)
        '''
        plot_roc_cross_val(self.X, self.Y, models)
