import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency
from .plots import plot_corr, plot_mainfold, plot_numerical, plot_categorical


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
        self.target = target
        self.targetname = targetname
        if (filename and target):
            self.read_csv(target, filename)

    def set_target(self, target, targetname=None):
        ''' Sets the target variable and if the data value exists,
        the X and Y values are setted as well

        :param target: Categorical variable to classify
        :type target: str
        :param targetname: Target name for reports
        :type targetname: str or None
        '''
        self.target = target
        self.targetname = self.targetname or targetname
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
        self.set_target(target)
        return self.X, self.Y, self.data

    def standarize(self, normalizer, scaler):
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

    def split_classes(self, label):
        ''' Returns the splited value of the dataset using the requested label

        :param label: Name of the variable to split
        :type label: str
        :return: The `positive` and `negative` data splited
        :rtype: tuple(pd.Series, pd.Series)
        '''
        positive = self.X.loc[self.data[self.target] == 1][label]
        negative = self.X.loc[self.data[self.target] != 1][label]
        return positive, negative

    def features_metrics(self, plot=None):
        ''' Checks for each variable the probability of being splited

        :param plot: Plots the variables, showing the difference in the classes
        :type plot: 'all' or 'categorical' or 'numerical' or None
        :return: Table of variables and their classification tests
        :rtype: pd.DataFrame
        '''
        plot_cat = plot in ['all', 'categorical']
        plot_num = plot in ['all', 'numerical']

        features = {}
        features_cols = ['cardinality kind', 'split probability']
        for column in self.X.columns.tolist():
            # Categorical case
            if len(self.X[column].unique().tolist()) <= 2:
                if plot_cat:
                    plot_categorical(self.X, self.Y, column, self.targetname)
                cont_table = pd.crosstab(self.Y, self.X[column], margins=False)
                test = chi2_contingency(cont_table.values)
                features[column] = ['categorical', f'{(test[1] * 100):.4f}%']
            # Numerical case
            else:
                positive, negative = self.split_classes(column)
                if plot_num:
                    plot_numerical(positive, negative, column, self.targetname)
                _, p_value = ttest_ind(positive, negative)
                features[column] = ['numerical', f'{(p_value * 100):.4f}%']
        return pd.DataFrame(features, index=features_cols)

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
