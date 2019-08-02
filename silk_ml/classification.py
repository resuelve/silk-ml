import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency


class Classifier:
    ''' General tasks for classification and data analysis

    :param target: categorical variable to classify
    :type target: string
    :param filename: name with path for reading a csv file
    :type filename: string

    Example::
        cl1 = Classifier()
        cl2 = Classifier('test')
        cl3 = Classifier(target='test')
        cl4 = Classifier('test', 'path/file')
        cl5 = Classifier(target='test', filename='path/file')
    '''

    def __init__(self, target=None, filename=None):
        self.target = target
        if (filename and target):
            self.read_csv(target, filename)

    def set_target(self, target):
        ''' Sets the target variable and if the data value exists,
        the X and Y values are setted as well

        :param target: categorical variable to classify
        :type target: string
        '''
        self.target = target
        if self.data is not None:
            self.Y = self.data[target]
            self.X = self.data.drop(columns=[target])

    def read_csv(self, target, filename):
        ''' Reads a CSV file and separate the X and Y variables

        :param target: categorical variable to classify
        :type target: string
        :param filename: name with path for reading a csv file
        :type filename: string
        :return: `X`, `Y`, and `data` values
        :rtype: list(DataFrame)
        '''
        self.data = pd.read_csv(filename)
        self.set_target(target)
        return self.X, self.Y, self.data

    def standarize(self, normalizer, scaler):
        ''' Applies a normalizer and scaler preprocessing steps

        :param normalizer: class that centers the data
        :type normalizer: implements `fit_transform` method
        :param scaler: class that modifies the data boundaries
        :type scaler: implements `fit_transform` method
        '''
        normalized = normalizer.fit_transform(self.X).transpose()

        # Check if in the normalization any data get lost
        for i, column in enumerate(self.X.columns.tolist()):
            if normalized[i].var() <= 1e-10:
                normalized[i] = self.X[column]

        standard = scaler.fit_transform(normalized.transpose())
        self.X.loc[:, :] = standard

    def split_classes(self, label):
        ''' Returns the splited value of the dataset using the requested label
        
        :return: the `positive` and `negative` data splited
        :rtype: (list, list)
        '''
        positive = self.X.loc[self.data[self.target] == 1][label]
        negative = self.X.loc[self.data[self.target] != 1][label]
        return positive, negative

    def features_metrics(self):
        ''' Checks for each variable the probability of being splited

        :return: table of variables and their classification tests
        :rtype: DataFrame
        '''
        features = {}
        features_cols = ['cardinality kind', 'split probability']
        for column in self.X.columns.tolist():
            # Categorical case
            if len(self.X[column].unique().tolist()) <= 2:
                cont_table = pd.crosstab(self.Y, self.X[column], margins=False)
                test = chi2_contingency(cont_table.values)
                features[column] = ['categorical', f'{(test[1] * 100):.4f}%']
            # Numerical case
            else:
                positive, negative = self.split_classes(column)
                _, p_value = ttest_ind(positive, negative)
                features[column] = ['numerical', f'{(p_value * 100):.4f}%']
        return pd.DataFrame(features, index=features_cols)
