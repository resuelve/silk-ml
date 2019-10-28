from sklearn.metrics import confusion_matrix


def ls_score(y, y_predicted):
    """ Score that punishes the false negative values, that goes from -1 to 1
    Args:
        y (list): 1d array-like, or label indicator array / sparse matrix
            ground truth (correct) labels.
        y_predicted (list):1d array-like, or label indicator array / sparse
            matrix predicted labels, as returned by a classifier.

    Returns:
        float: A score between -1 and 1 that indicates the correctness of the
            classification
    """
    conf_matrix = confusion_matrix(y, y_predicted)
    assert conf_matrix.shape == (2, 2)
    [tn, fp], [fn, tp] = conf_matrix
    assert (tp + fn) != 0
    assert (tn + fp) != 0
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    return tpr * (tnr + 1) - 1
