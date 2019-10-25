from .classification import Classifier
from .plots import plot_corr, plot_mainfold, plot_categorical, plot_numerical
from .plots import plot_roc_cross_val
from .features import split_classes, features_metrics
from .imbalanced import resample
from .train import cross_validation
from .scores import ls_score

__name__ = 'silk_ml'
__version__ = '0.1.1'
__all__ = ['classification', 'plots', 'features', 'imbalanced', 'train']
