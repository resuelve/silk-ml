import logging
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, f_regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier

def kbest(xcols, X, y, sf=f_classif, k=10):
    """
    Se seleccionan las k variables con mejor relación con la respuesta a través
    de pruebas estadísticas

    Args:
        xcols (array): Con los nombres de las variables en X
        X (array): Matriz de inputs
        y (array): Vector de variable objetivo
        sf (function): Prueba estadística (f_classif, f_regression)
        k (int): Número de variables
    Returns:
        features (list): Lista con las k mejores variables
        fit (feature_selection): Feature selector (univariado)
    """
    test = SelectKBest(score_func=sf, k=k)
    fit = test.fit(X, y)
    sup = fit.get_support()
    features = list(xcols[sup])

    return features, fit

def rec_feat_elim(xcols, X, y, estimator=LogisticRegression(), k=10):
    """
    RFE (Recursive Feature Elimination) elimina atributos recursivamente
    y crea modelos con los que permanecen para escoger las
    k mejores variables con las que se quedará el modelo

    Args:
        xcols (array): Con los nombres de las variables en X
        X (array): Matriz de inputs
        y (array): Vector de variable objetivo
        estimator (model): Modelo de scikit-learn
                           (LogisticRegression(), LinearRegression())
        k (int): Número de variables
    Returns:
        features (list): Lista con las k mejores variables
        selector (feature_selection): Feature selector (RFE)
    """
    selector = RFE(estimator, n_features_to_select=k, step=1)
    selector = selector.fit(X, y)
    sup = selector.support_
    features = list(xcols[sup])

    return features, selector

def PCA_dec(X, n_comp=10):
    """
    Descomposición en componentes principales (Reducción de dimensionalidad)

    Args:
        X (array): Matriz de inputs
        n_comp (int): Número de componentes prncipales al que se quiere reducir
                      la dimensión
    Returns:
        fit (decomposition): scikit-learn PCA decomposition
    """

    pca = PCA(n_components=3)
    fit = pca.fit(X)

    return fit

def get_vif(df):
    """
    Nos da el factor de inflación de la varianza de cada variable independiente

    Args:
        df (DataFrame): DataFrame con datos de nuestras variables independientes
    Returns:
        vif (DataFrame): DataFrame con el factor de inflación de la varianza
                         de cada variable
    """
    vif = pd.DataFrame()
    X = df.drop(response, 1)
    X['intercept'] = 1
    x = X.values
    vif['vif'] = [variance_inflation_factor(x, i) for i in range(x.shape[1])]
    vif['feature'] = X.columns

    return vif

def importance_corr(df, response, corr=0.1, fif=0.01, vif=False):
    """
    Extra Trees Classifier (Extremely Randomized Trees) crea divisiones en
    los atributos con árboles de decisión para darles un valor de importancia,
    también se da la correlación y el factor de inflación de varianza

    Args:
        df (DataFrame): DataFrame con todos los datos
        response (str): Variable objetivo
        vif (boolean): Si queremos factor de inflación de varianza
    Returns:
        best_features (DataFrame): Con variables, correlaciones e importancia
    """
    X = df.drop(response, 1).values
    y = df[response].values
    etc = ExtraTreesClassifier()
    etc.fit(X, y)
    correl = pd.DataFrame(df.corr()[response])
    correl = correl.reset_index()
    correl.columns = ['feature', 'correlation']
    correl = correl[correl['feature'] != response]
    feature_importance = pd.DataFrame(df.columns, columns=['feature'])
    feature_importance = feature_importance[feature_importance['feature'] \
                                                              != response]
    feature_importance['importance'] = etc.feature_importances_
    importance_correl = pd.merge(feature_importance, correl, on='feature')
    if vif != False:
        VF = get_vif(df)
        importance_correl = pd.merge(importance_correl, VF, on= 'feature')

    leakage = importance_correl[(abs(importance_correl['correlation']) >= 0.5) \
                              | (abs(importance_correl['importance']) >= 0.01)]
    best_features = importance_correl[(abs(importance_correl['correlation']) \
                     >= corr) | (abs(importance_correl['importance']) >= fif)]
    logging.info('Hay ' + str(len(leakage)) + \
                 ' variables que pueden presentar data leakage\n')
    for i in leakage['feature'].values:
        logging.info('Variable: ' + i)
        logging.info('puede presentar leakage, desea eliminarla? (si o no)')
        answer = input()
        if answer == 'si':
            best_features = best_features[best_features['feature'] != i]

    best_features = best_features.reset_index(drop=True)
    logging.info('''\nEstas son las variables que estaremos usando, si
    desea eliminar alguna escriba el número que aparece a la izquierda de
    las variables a eliminar, separados por comas''')
    pd.set_option('display.max_rows', len(best_features))
    display(best_features)

    elim = input()
    if elim != '':
        elim = elim.replace(' ', '')
        elim = elim.split(',')
        ranges = [i for i in elim if '-' in i]
        nonranges = [int(i) for i in elim if i not in ranges]
        if ranges:
            r = []
            for x in ranges:
                rn = list(range(int(x.split('-')[0]), int(x.split('-')[1])))
                r.extend(rn)
            r.append(r[-1] + 1)
            nonranges.extend(r)
        best_features = best_features.drop(nonranges)

    return best_features
