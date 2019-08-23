import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_corr(data, values=True):
    ''' Plots correlation matrix

    :param data: Data to compute correlation matrix
    :type data: pd.DataFrame
    :param data: Data to compute correlation matrix
    :type data: bool or None
    '''
    corr = data.corr()
    _fig, ax = plt.subplots(figsize=(50 if values else 20, 10))
    sns.set(style='white')
    sns.heatmap(corr, annot=values, fmt='.3', linewidths=.5, ax=ax, center=0)


def plot_mainfold(method, data, target_name):
    ''' Plots the information using dimentionality reduction

    :param method: Mainfold transformation method
    :type method: Class.fit_transform
    :param data: Dataset to reduce, with two classes
    :type data: pd.DataFrame
    '''
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
    ''' Plots the categorical variable, showing the two classes

    :param X: Main dataset with the categorical variables
    :type X: pd.DataFrame
    :param Y: Target variable
    :type Y: pd.Series
    :param catego_var: Name of the categorical variable to plot
    :type catego_var: str
    :param target_name: Name of the target variable to classify
    :type target_name: str
    '''
    X_copy = X.copy()
    X_copy[target_name] = pd.Series(Y).map(
        lambda x: target_name if x == 1 else f'not {target_name}'
    )
    sns.countplot(x=target_name, hue=catego_var, data=X_copy)
    plt.show()


def plot_numerical(positive, negative, numeric_var, target_name):
    ''' Plots the information using dimentionality reduction

    :param positive: Serie with the positive class to plot
    :type positive: pd.Series
    :param negative: Serie with the negative class to plot
    :type negative: pd.Series
    :param numeric_var: Name of the numerical variable to plot
    :type numeric_var: str
    :param target_name: Name of the target variable to classify
    :type target_name: str
    '''
    plt.hist(positive, bins=25, alpha=0.6, label=target_name)
    plt.hist(negative, bins=25, alpha=0.6, label=f'not {target_name}')
    plt.xlabel(numeric_var, fontsize=12)
    plt.legend(loc='upper right')
    plt.show()
