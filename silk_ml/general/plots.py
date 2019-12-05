import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold


def plot_corr(data, values=True):
    """ Plots correlation matrix

    Args:
        data (pd.DataFrame): Data to compute correlation matrix
        values (bool or None): Plot values in the matrix
    """
    corr = data.corr()
    _fig, ax = plt.subplots(figsize=(50 if values else 20, 10))
    sns.set(style='white')
    sns.heatmap(corr, annot=values, fmt='.3', linewidths=.5, ax=ax, center=0)


def plot_mainfold(method, data, target_name):
    """ Plots the information using dimensionality reduction

    Args:
        method (Class.fit_transform): Mainfold transformation method
        data (pd.DataFrame): Dataset to reduce, with two classes
        target_name (str): Name of the variable to classify
    """
    data_compacted = method.fit_transform(data)
    _fig, ax = plt.subplots()
    win_x = []
    win_y = []
    lose_x = []
    lose_y = []
    for i, x in enumerate(data_compacted):
        if data[target_name][i] == 0:
            win_x.append(x[0])
            win_y.append(x[1])
        else:
            lose_x.append(x[0])
            lose_y.append(x[1])

    ax.scatter(win_x, win_y, c='blue', alpha=0.3, edgecolors='none',
               label=f'{target_name} ({len(win_x)})')
    ax.scatter(lose_x, lose_y, c='red', alpha=0.3, edgecolors='none',
               label=f'not {target_name} ({len(lose_x)})')

    ax.legend()
    ax.grid(True)
    plt.show()


def plot_categorical(X, Y, catego_var, target_name):
    """ Plots the categorical variable, showing the two classes

    Args:
        X (pd.DataFrame): Main dataset with the categorical variables
        Y (pd.Series): Target variable
        catego_var (str): Name of the categorical variable to plot
        target_name (str): Name of the target variable to classify
    """
    X_copy = X.copy()
    X_copy[target_name] = pd.Series(Y).map(
        lambda x: target_name if x == 1 else f'not {target_name}'
    )
    sns.countplot(x=target_name, hue=catego_var, data=X_copy)
    plt.show()


def plot_numerical(positive, negative, numeric_var, target_name):
    """ Plots the information using dimentionality reduction

    Args:
        positive (pd.Series): Serie with the positive class to plot
        negative (pd.Series): Serie with the negative class to plot
        numeric_var (str): Name of the numerical variable to plot
        target_name (str): Name of the target variable to classify
    """
    plt.hist(positive, bins=25, alpha=0.6, label=target_name)
    plt.hist(negative, bins=25, alpha=0.6, label=f'not {target_name}')
    plt.xlabel(numeric_var, fontsize=12)
    plt.legend(loc='upper right')
    plt.show()


def single_cross_val(classifier, model_name, color, X, Y):
    """ Appends a ROC from the classifier

    Args:
        classifier: Model to run the classification task to append to the plot
        model_name (str): Name of the model for the plot
        color (str): Color to plot
        X (pd.DataFrame): Main dataset with the variables
        Y (pd.Series): Target variable
    """
    cross_val = StratifiedKFold(n_splits=6)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    X = np.array(X)
    Y = np.array(Y)

    for train, test in cross_val.split(X, Y):
        probas = classifier.fit(X[train], Y[train]).predict_proba(X[test])
        # Computa ROC
        fpr, tpr, _ = roc_curve(Y[test], probas[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    label = f'{model_name} (AUC = {mean_auc:.2f} +/- {std_auc:.2f})'
    plt.plot(mean_fpr, mean_tpr, color=color, lw=2, alpha=0.8, label=label)


def plot_roc_cross_val(X, Y, models):
    """ Plots all the models with their ROC

    Args:
        X (pd.DataFrame): Main dataset with the variables
        Y (pd.Series): Target variable
        models (list(tuple)): Models to evaluate
    """
    color_map = plt.cm.get_cmap('hsv', len(models))
    for i, (model_name, model) in enumerate(models):
        single_cross_val(model, model_name, color_map(i), X, Y)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='0.75', alpha=0.8,
             label='Baseline')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc='lower right')
    plt.show()
