import math
import logging
import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from .replace_nans import replace_nan
from silk_ml.data_cleaning.cleaning import datatypes

logging.getLogger().setLevel(logging.DEBUG)


def check_correlation(df_pair, threshold=0.5):
    """
    Checa la correlación de un par de columnas de un DataFrame
    Args:
		df_pair (Datframe): Dataframe con la primer columna a comparar por la
				segunda
		threshold:
    Returns:
		result (list) : lista con el valor de la variable y su correlación
    """
    varname = df_pair.columns[0]
    response = df_pair.columns[1]
    correlation = df_pair.corr()[response][0]
    if abs(correlation) < abs(threshold) or math.isnan(correlation):
        result = varname
    else:
        result = ''

    return result


def drop_correlation(DF, var_list, response, threshold=0.1):
    """
    De una lista de variables quita las que tengan menor correlación del
    DataFrame
    Args:
            DF (Dataframe): Dataframe con los datos aumentados
            var_list (list): lista de variables a verificar correlación
            response (string): nombre de la variable dependiente a predecir
    Response:
            DF (Datframe): Dataframe con variables que tienen mayor relación con
                    la variable a predecir
            dropped (list): lista de variables desechadas de la lista por baja
                    correlación
    """
    logging.info("*** Checando correlación de variables nuevas contra {} \
				(puede tardar algunos minutos)***".format(response))
    df = DF.copy()
    # Correlación con cada variable de la lista contra la variable dependiente
    pair_columns = [[element, response] for element in var_list]

    df_pairs = [df[pair] for pair in pair_columns]

    pool = mp.Pool(processes=4)

    correlations_drop = pool.imap_unordered(check_correlation, df_pairs)
    pool.close()
    pool.join()

    correlations_drop = [varname for varname in correlations_drop if
                         varname not in ('')]
    # eliminamos la columna que no cumple con correlación mínima
    logging.info("eliminamos ({}/{}) variables que no tienen buena correlación \
			con {}".format(len(correlations_drop), len(df.columns),
                  response))
    df = df.drop(correlations_drop, 1)
    return df, correlations_drop


def augment_numeric(DF, response):
    """
Crea una lista con transformación de variables numéricas del DataFrame
Args:
    DF (Dataframe): Dataframe con los datos numéricos a aumentar
    response (list): nombre de las variables a predecir
Response:
    DF (DataFrame): Dataframe con los datos numéricos aumentados
    new_vars (list): lista con nombre de variables aumentadas candidatas
"""
    df = DF.copy()

    numericas = list(df.select_dtypes(include=['int', 'float']).columns)

    df = replace_nan(df, numericas)  # cambiar a función que predice valores

    numericas = list(filter(lambda x: x not in response, numericas))
    new_vars = []
    for i in numericas:
        try:
            varname = i + '^' + str(2)
            df[varname] = df[i] ** 2
            new_vars.append(varname)
            varname = i + '^' + str(3)
            df[varname] = df[i] ** 3
            new_vars.append(varname)
            varname = 'sqrt(' + i + ')'
            df[varname] = np.sqrt(df[i])
            new_vars.append(varname)
            varname = '1/' + i
            df[varname] = 1 / df[i]
            new_vars.append(varname)
            varname = 'log(' + i + ')'
            df[varname] = df[i].apply(np.log)
            new_vars.append(varname)
            df = df.replace(-np.inf, -1000)
            df = df.replace(np.inf, 1000)
        except Exception as e:
            logging.error(e)

    df.drop(columns='intercept', axis=1, inplace=True)

    return df, new_vars


def augment_date(DF, response):
    """
Crea una lista con transformación de variables de fechas del DataFrame
Args:
    DF (Datframe): Datframe con los datos de fecha a aumentar
    response (list): nombre de las variables dependiente a predecir
Response:
    DF (Dataframe): Dataframe con los datos de fechas aumentados
    new_vars(list): lista con las variables aumentadas candidatas
"""
    df = DF.copy()
    fechas = list(df.select_dtypes(include=['datetime',
                                            'datetime64[ns]']).columns)
    fechas = list(filter(lambda x: x not in response, fechas))
    original_cols = list(df.columns)
    newvars = []
    unuseful = []
    acum_fechas = []
    new_vars = []
    for i in fechas:
        varname = 'hora_' + i
        df[varname] = df[i].dt.hour
        new_vars.append(varname)
        varname = 'dia_' + i
        df[varname] = df[i].dt.day
        new_vars.append(varname)
        varname = 'mes_' + i
        df[varname] = df[i].dt.month
        new_vars.append(varname)
        varname = 'dia_semana_' + i
        lista_de_dias_semana = []
        for ejemplo in df[i]:
            if math.isnan(ejemplo.weekday()):
                lista_de_dias_semana.append(0)
            else:
                lista_de_dias_semana.append(int(ejemplo.weekday()))
        df[varname] = lista_de_dias_semana
        new_vars.append(varname)
        acum_fechas.append(i)
        for j in [x for x in fechas if x not in acum_fechas]:
            # Diferencia de fechas (en días)
            varname = i + '-' + j
            df.loc[(df[i].notnull()) & (df[j].notnull()), varname] = (
                df[i] - df[j]).dt.days
            new_vars.append(varname)

    df = pd.get_dummies(df, columns=new_vars)
    df.drop(fechas, 1)
    new_vars = [i for i in df.columns if i not in original_cols]
    return df, new_vars


def augment_categories(DF, response):
    """
	Se hacen transformaciones con operaciones lógicas entre variables categóricas
	dentro de un Dataframe
	Args:
		DF (DataFrame): Dataframe de donde se quieren aumentar las categorías
		response(list): lista con nombres de las variables dependientes a predecir
	Response:
		df (Dataframe): DataFrame con las variables categóricas aumentadas
		new_vars(list): lista con las variables categóricas candidatas aumentadas
	"""
    df = DF.copy()
    dummy_vars = []
    for i in df.columns:
        if set(df[i].unique()) == {0, 1}:
            dummy_vars.append(i)

    dummy_vars = list(filter(lambda x: x not in response, dummy_vars))
    new_vars = []

    logging.info("*** Aumentando {} categorías ***".format(len(dummy_vars)))
    pbar = tqdm(total=len(dummy_vars))
    for i in dummy_vars:
        for j in [x for x in dummy_vars if x not in new_vars]:
            # Multiplicación de conectores lógicos (AND, OR, NAND, NOR, XOR & XNOR)
            varname = i + '*' + j
            df[varname] = df[i].astype(int) & df[j].astype(int)
            new_vars.append(varname)

            varname = i + '+' + j
            df[varname] = df[i].astype(int) | df[j].astype(int)
            new_vars.append(varname)

            varname = 'nand(' + i + ',' + j + ')'
            new_vars.append(varname)
            df[varname] = ~(df[i].astype(int) & df[j].astype(int))

            varname = 'nor(' + i + ',' + j + ')'
            new_vars.append(varname)
            df[varname] = ~(df[i].astype(int) | df[j].astype(int))

            varname = 'xor(' + i + ',' + j + ')'
            new_vars.append(varname)
            df[varname] = (df[i].astype(int) & ~(df[j].astype(int))) | \
                (~(df[i].astype(int)) & df[j].astype(int))

            varname = 'xnor(' + i + ',' + j + ')'
            new_vars.append(varname)
            df[varname] = (df[i].astype(int) & df[j].astype(int)) | \
                (~(df[i].astype(int)) & ~(df[j].astype(int)))
        pbar.update(1)
    pbar.close()
    # Algunas operaciones se quedan en valores lógicos. Hay que pasarlas a binarias

    logging.info("*** Verificando datos de {} nuevas variables ***".format(
        len(new_vars)))
    pbar = tqdm(total=len(new_vars))
    for var in new_vars:  # TODO: hacer este chequeo cada que se añade una variable
        df.loc[df[var] == True, var] = 1
        df.loc[df[var] == False, var] = 0
        pbar.update(1)
    pbar.close()

    return df, new_vars


def augment_data(DF, response, threshold=0.1, categories=False,
                 exclude_metadata=True):
    """
	Prueba ciertas transformaciones numéricas, de fecha y categóricas.
	Verifica si la correlación es buena, a partir de un threshold,
	para agregarlas al dataframe resultante
	Args:
		DF (DataFrame): DataFrame de tus datos
		response (str): Variable dependiente (la debe contener tu base)
		threshold (float): Correlación mínima que se espera de una variable que
						quieres que entre al modelo
	Returns:
		df (DataFrame): DataFrame con transformaciones útiles
		new_vals (list): Lista de variables transformadas nuevas en el dataframe
	"""

    logging.info('***Haciendo agregación de datos***')
    df = DF.copy()

    if exclude_metadata:
        metadata_vars = [var for var in df.columns if var.startswith('__')]
        df.drop(metadata_vars, inplace=True, axis=1)
    catego = []
    df, numeric = augment_numeric(df, response)
    df, fecha = augment_date(df, response)
    # suele tardarse mucho la transformación de categorías
    numericas, categoricas, fechas = datatypes(df)
    df = pd.get_dummies(df, columns=categoricas,
                        dummy_na=True, drop_first=True)
    if categories:
        df, catego = augment_categories(df, response)
    aug_vars = numeric + fecha + catego
    df.drop(fechas, inplace=True, axis=1)
    df, dropped = drop_correlation(df, aug_vars, response, threshold)

    return df, dropped
