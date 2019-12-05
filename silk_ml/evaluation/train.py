import pandas as pd
from sklearn.model_selection import cross_validate


def cross_validation(X, Y, models, scores, folds=30):
    """ Validates several models and scores

    Args:
        X (pd.DataFrame): Main dataset with the variables
        Y (pd.Series): Target variable
        models (list(tuple)): Models to evaluate
        scores (list(tuple)): Scores to measure the models
        folds (int): Number of folds in a (Stratified)KFold
    """
    score_table = {}
    for model_name, model in models:
        scores_res = cross_validate(model, X, Y, cv=folds, scoring=scores)
        score_table[model_name] = []
        for name in scores.keys():
            res = scores_res[f'test_{name}']
            val = f'{res.mean():.4f} (+/- {res.std()*2:.4f})'
            score_table[model_name].append(val)
    return pd.DataFrame(score_table, index=list(scores.keys()))
