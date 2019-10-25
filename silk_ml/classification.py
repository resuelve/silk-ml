import pandas as pd

from .features import features_metrics
from .plots import plot_corr, plot_mainfold, plot_roc_cross_val
from .train import cross_validation
from .imbalanced import resample


class Classifier:
    """ General tasks for classification and data analysis

    Args:
        target (str or None): Categorical variable to classify
        filename (str or None): Name with path for reading a csv file
        target_name (str or None): Target name for reports
    """

    def __init__(self, target=None, filename=None, target_name=None):
        pd.set_option('display.max_columns', None)
        self._target = target
        self.target_name = target_name
        if filename and target:
            self.read_csv(target, filename)

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, target):
        """ Sets the target variable and if the data value exists,
        the X and Y values are setted as well

        Args:
            target (str): Categorical variable to classify
        """
        self._target = target
        if self.data is not None:
            self.Y = self.data[target]
            self.X = self.data.drop(columns=[target])

    def read_csv(self, target, filename):
        """ Reads a CSV file and separate the X and Y variables

        Args:
            target (str): Categorical variable to classify
            filename (str): Name with path for reading a csv file

        Returns:
            list(pd.DataFrame): `X`, `Y`, and `data` values
        """
        self.data = pd.read_csv(filename)
        self.target = target
        return self.X, self.Y, self.data

    def standardize(self, normalizer, scaler):
        """ Applies a normalizer and scaler preprocessing steps

        Args:
            normalizer (Class.fit_transform): Class that centers the data
            scaler (Class.fit_transform): Class that modifies the data boundaries
        """
        normalized = normalizer.fit_transform(self.X).transpose()

        # Check if in the normalization any data get lost
        for i, column in enumerate(self.X.columns.tolist()):
            if normalized[i].var() <= 1e-10:
                normalized[i] = self.X[column]

        return scaler.fit_transform(normalized.transpose())

    def features_metrics(self, plot=None):
        """ Checks for each variable the probability of being splited

        Args:
            plot ('all' or 'categorical' or 'numerical' or None): Plots the
                variables, showing the difference in the classes

        Returns:
            pd.DataFrame: Table of variables and their classification tests
        """
        return features_metrics(self.X, self.Y, self.target_name, plot)

    def remove_features(self, features):
        """ Remove features from the X values

        Args:
            features (list(str)): Column's names to remove
        """
        self.X = self.X.drop(columns=features)

    def resample(self, rate=0.9, strategy='hybrid'):
        """ Sampling based methods to balance dataset

        Args:
            rate (float): Ratio of the number of samples in the minority class
                over the number of samples in the majority class after
                resampling
            strategy ('hybrid' or 'over_sampling' or 'under_sampling'): Strategy
                to balance the dataset
        """
        self.X, self.Y = resample(self.X, self.Y, rate, strategy)

    def cross_validation(self, models, scores, folds=30):
        """ Validates several models and scores

        Args:
            models (list(tuple)): Models to evaluate
            scores (list(tuple)): Scores to measure the models
            folds (int): Number of folds in a (Stratified)KFold
        """
        return cross_validation(self.X, self.Y, models, scores, folds)

    def plot_corr(self, values=True):
        """ Plots the correlation matrix

        Args:
            values (bool): Shows each of the correlation values
        """
        plot_corr(self.data, values)

    def plot_mainfold(self, method):
        """ Plots the reduced space using a mainfold transformation

        Args:
            method (Class.fit_transform): Mainfold transformation method
        """
        plot_mainfold(method, self.data, self.target_name)

    def plot_roc_cross_val(self, models):
        """ Plots all the models with their ROC

        Args:
            models (list(tuple)): Models to evaluate
        """
        plot_roc_cross_val(self.X, self.Y, models)
