import pandas as pd
from sklearn.model_selection import cross_validate

def cross_validation(X, Y, models, scores, iterations=30):
    ''' Validates several models and scores

    :param X: Main dataset with the variables
    :type X: pd.DataFrame
    :param Y: Target variable
    :type Y: pd.Series
    :param models: Models to evaluate
    :type models: list(tuple)
    :param scores: Scores to measure the models
    :type scores: list(tuple)
    :param iterations: Number of iteration over the cross validation
    :type iterations: int
    '''
    score_table = {}
    for model_name, model in models:
        scores_res = cross_validate(model, X, Y, cv=iterations, scoring=scores)
        score_table[model_name] = []
        for name in scores.keys():
            res = scores_res[f'test_{name}']
            val = f'{res.mean():.4f} (+/- {res.std()*2:.4f})'
            score_table[model_name].append(val)
    return pd.DataFrame(score_table, index=list(scores.keys()))
