import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency


class Classificator:
    def __init__(self, target=None, filename=None):
        self.target = target
        if (filename and target):
            self.read_csv(target, filename)

    def set_target(self, target):
        self.target = target
        if self.data is not None:
            self.Y = self.data[target]
            self.X = self.data.drop(columns=[target])

    def read_csv(self, target, filename):
        self.data = pd.read_csv(filename)
        self.set_target(target)
        return self.X, self.Y, self.data

    def standarize(self, normalizer, scaler):
        normalized = normalizer.fit_transform(self.X).transpose()

        # Check if in the normalization any data get lost
        for i, column in enumerate(self.X.columns.tolist()):
            if normalized[i].var() <= 1e-10:
                normalized[i] = self.X[column]

        standard = scaler.fit_transform(normalized.transpose())
        self.X.loc[:, :] = standard

    def split_classes(self, label):
        positive = self.data.loc[self.data[self.target] == 1][label]
        negative = self.data.loc[self.data[self.target] != 1][label]
        return positive, negative

    def features_metrics(self):
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
