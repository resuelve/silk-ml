import statsmodels.api as sm
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier, TPOTRegressor
from sklearn import preprocessing


def train_test(df, response, train_size=0.75, time_series=False, scaling=None):
    """
	Regresa train y test sets

	Args:
		df (DataFrame): Datos listos para el modelo
		response (str): Variable respuesta
		train_size (float): % Train Size
		time_series (boolean): Si es serie de tiempo o no
		scaling (str): ['standard', 'minmax', 'maxabs', 'robust', 'quantile']
	Returns:
		x_train (Array): conjunto de datos de entrenamiento (indep)
		x_test (Array): conjunto de datos de prueba (indep)
		y_train (Array): conjunto de datos de entrenamiento (dep)
		y_test (Array): conjunto de datos de prueba (dep)
	"""

    data = df.copy()
    X = data.drop(response, 1)
    y = data[response]

    logging.info('X columns')
    logging.info(list(X.columns))
    logging.info('Response')
    logging.info(response)

    if time_series:
        train_size = int(train_size * len(X))
        x_train = X[:train_size].values
        x_test = X[train_size:].values
        y_train = y[:train_size].values
        y_test = y[train_size:].values

    else:
        x_train, x_test, y_train, y_test = train_test_split(X.values,
                                                            y.values,
                                                            random_state=0,
                                                            train_size=train_size)
    if scaling == 'standard':
        scaler = preprocessing.StandardScaler()
    if scaling == 'minmax':
        scaler = preprocessing.MinMaxScaler()
    if scaling == 'maxabs':
        scaler = preprocessing.MaxAbsScaler()
    if scaling == 'robust':
        scaler = preprocessing.RobustScaler()
    if scaling == 'quantile':
        scaler = preprocessing.QuantileTransformer()

    if scaling is not None:
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test


def tpotclass(X_train, y_train):
    """
	Usando TPOT (Tree-Based Pipeline Optimization Tool), librería de AutoML,
	genera el "mejor" modelo de clasificación automáticamente
	Args:
		X_train (Array): conjunto de datos de entrenamiento (regresores)
		y_train (Array): conjunto de datos de entrenamiento (objetivo)
	returns:
		tpotmod (modelo): Modelo de clasificación generado con TPOT
	"""
    pipeline_optimizer = TPOTClassifier(generations=5,
                                        population_size=50,
                                        cv=5,
                                        random_state=42,
                                        verbosity=2,
                                        n_jobs=4)
    tpotmod = pipeline_optimizer.fit(X_train, y_train)

    return tpotmod


def tpotreg(X_train, y_train):
    """
	Usando TPOT (Tree-Based Pipeline Optimization Tool), librería de AutoML,
	genera el "mejor" modelo de regresión automáticamente
	Args:
		X_train (Array): conjunto de datos de entrenamiento (regresores)
		y_train (Array): conjunto de datos de entrenamiento (objetivo)
	returns:
		tpotmod (modelo): Modelo de regresión generado con TPOT
	"""

    pipeline_optimizer = TPOTRegressor(generations=5,
                                       population_size=50,
                                       cv=5,
                                       random_state=42,
                                       verbosity=2,
                                       n_jobs=4)
    tpotmod = pipeline_optimizer.fit(X_train, y_train)

    return tpotmod


def logreg(X_train, y_train):
    """
	Calcula modelo de Regresión Logística
	Args:
		X_train (Array): conjunto de datos de entrenamiento (regresores)
		y_train (Array): conjunto de datos de entrenamiento (objetivo)
	returns:
		logreg (modelo): Regresión Logística
	"""
    try:
        # Si la matriz es singular va a dar error
        log = sm.Logit(y_train, X_train)
        logreg_model = log.fit()
    except Exception as e:
        # Intentamos con la matriz hessiana
        print(e)
        log = sm.Logit(y_train, X_train)
        logreg_model = log.fit(method='bfgs')

    return logreg_model


def linreg(X_train, y_train):
    """
	Calcula modelo de Regresión Lineal
	Args:
		X_train (Array): conjunto de datos de entrenamiento (regresores)
		y_train (Array): conjunto de datos de entrenamiento (objetivo)
	returns:
		linreg_model (modelo): Regresión Lineal
	"""
    linreg = sm.OLS(y_train, X_train)
    linreg_model = linreg.fit()

    return linreg_model


def simple_model(X_train, y_train, tpot=False):
    """
	Obtiene variable objetivo, decide si es de clasificación o regresión
	y regresa un modelo simple
	Args:
		X_train (Array): conjunto de datos de entrenamiento (regresores)
		y_train (Array): conjunto de datos de entrenamiento (objetivo)
		tpot (boolean): si queremos generar modelo con tpot
	returns:
		model (modelo): Regresión Logística o Lineal dependiendo de la variable
						objetivo
		tpotmod (modelo): Modelo de Regresión o Clasificación generado con TPOT
	"""
    tpotm = None
    # Revisamos si es modelo de clasificación binaria
    if len(set(np.unique(y_train))) == 2:
        model = logreg(X_train, y_train)
        if tpot:
            toptm = tpotclass(X_train, y_train)
    elif len(set(np.unique(y_train))) > 2 and len(set(np.unique(y_train))) < 10:
        multilog = sm.MNLogit(y_train, X_train)
        model = multilog.fit()
        if tpot:
            tpotm = tpotclass(X_train, y_train)
    else:
        model = linreg(X_train, y_train)
        if tpot:
            tpotm = tpotreg(X_train, y_train)

    return model, tpotm
