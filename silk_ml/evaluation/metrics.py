import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature


def model_precision(y, predictions, sc, disp=False):
    """
	Métricas de precisión de un modelo de clasificación

	Args:
		y (array): Instancias de la variable dependiente
		predictions (array): Predicciones
		sc (float): Score de corte entre 0 y 1 que marca el límite de clasificación
					(arriba de sc se considera positivo)
		disp (boolean): Imprimir matriz con métricas
	Returns:
		accuracy (float): (tp+tn)/(tp+tn+fp+fn)
		precision (float): tp/(tp+fp)
		recall (float): tp/(tp+fn)
		f1_score (float): 2/(1/Precision+1/Recall) Media armónica
						entre Precision y Recall
		mcc (float): Matthiews Correlation Coefficient
					(tp*tn-fp*fn)/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))

	"""
    yt = y.copy()
    predictionst = predictions.copy()

    yt = yt.reshape([yt.shape[0], 1])
    predictionst = predictionst.reshape([predictionst.shape[0], 1])

    test = np.concatenate((yt, predictionst), axis=1)

    tp = ((test[:, 0] == 1) & (test[:, 1] >= sc)).sum()
    fp = ((test[:, 0] == 0) & (test[:, 1] >= sc)).sum()
    tn = ((test[:, 0] == 0) & (test[:, 1] < sc)).sum()
    fn = ((test[:, 0] == 1) & (test[:, 1] < sc)).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 / (1 / precision + 1 / recall)
    mcc = (tp * tn - fp * fn) / (
        math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    res = pd.DataFrame(0, index=['Accuracy', 'Precision',
                                 'Recall', 'F1 Score',
                                 'MCC'], columns=['Score'])

    res.loc['Accuracy'] = accuracy
    res.loc['Precision'] = precision
    res.loc['Recall'] = recall
    res.loc['F1 Score'] = f1_score
    res.loc['MCC'] = mcc

    return accuracy, precision, recall, f1_score, mcc


def tenbin_cutscore(y, predictions):
    """
	Precision por intervalos de 10 en 10

	Args:
		y (array): Instancias de la variable dependiente
		predictions (array): Predicciones

	Returns:
		res (DataFrame): Métricas de precisión por intervalos

	"""
    yt = y.copy()
    predictionst = predictions.copy()

    scoresindex = ['0-10',
                   '10-20',
                   '20-30',
                   '30-40',
                   '40-50',
                   '50-60',
                   '60-70',
                   '70-80',
                   '80-90',
                   '90-100']
    scorescolumns = ['Total', 'Positives']
    res = pd.DataFrame(0, index=scoresindex, columns=scorescolumns)

    yt.shape = [yt.shape[0], 1]
    predictionst.shape = [predictionst.shape[0], 1]

    test = np.concatenate((yt, predictionst), axis=1)

    low = 0
    up = 0.1
    for i in scoresindex:
        res.loc[i]['Total'] = ((test[:, 1] >= low) & (test[:, 1] < up)).sum()
        res.loc[i]['Positives'] = ((test[:, 1] >= low) & (test[:, 1] < up) &
                                   (test[:, 0] == 1)).sum()
        low += 0.1
        up += 0.1
    res['Positive Rate'] = res['Positives'].div(res['Total']) * 100
    res['% of Total'] = res['Total'].div(res['Total'].sum()) * 100
    res['% of Positives'] = res['Positives'].div(res['Positives'].sum()) * 100

    return res


def qcut_precision(y, predictions, n):
    """
	Precision por cuantiles

	Args:
		y (array): Instancias de la variable dependiente
		predictions (array): Predicciones
		n (int): Número de cuantiles

	Returns:
		res (DataFrame): Métricas de precisión por cuantiles

	"""
    predictionst = predictions.copy()
    predictionst.shape = (predictionst.shape[0],)
    ylist = y.tolist()
    plist = predictionst.tolist()

    cuts = pd.DataFrame({'y': ylist, 'predictions': plist})
    qcuts = pd.qcut(cuts['predictions'], n, duplicates='drop')
    cuts['qcut'] = qcuts
    cuts['qcut'] = cuts['qcut'].astype(str)

    ones = []
    zeros = []
    mean_score = []
    total = []
    for i in cuts['qcut'].unique():
        ones.append(len(cuts[(cuts['qcut'] == i) & (cuts['y'] != 0)]))
        zeros.append(len(cuts[(cuts['qcut'] == i) & (cuts['y'] != 1)]))
        total.append(len(cuts[cuts['qcut'] == i]))
        mean_score.append(cuts[cuts['qcut'] == i]['predictions'].mean())

    res = pd.DataFrame({'Total': total,
                        'Positives': ones,
                        '#0s': zeros,
                        'Mean Score': mean_score})

    res['Positive Rate'] = res['Positives'].div(res['Total']) * 100

    tp = []
    fp = []
    tn = []
    fn = []
    for i in res['Mean Score'].unique():
        tp.append(len(cuts[(cuts['predictions'] >= i) & (cuts['y'] != 0)]))
        fp.append(len(cuts[(cuts['predictions'] >= i) & (cuts['y'] != 1)]))
        tn.append(len(cuts[(cuts['predictions'] < i) & (cuts['y'] != 1)]))
        fn.append(len(cuts[(cuts['predictions'] < i) & (cuts['y'] != 0)]))

    res['TP'] = tp
    res['FP'] = fp
    res['TN'] = tn
    res['FN'] = fn

    res['Accuracy'] = (res['TP'] + res['TN']).div(res['TP'] + res['TN'] +
                                                  res['FP'] + res['FN'])
    res['Precision'] = res['TP'].div(res['TP'] + res['FP'])
    res['Recall'] = res['TP'].div(res['TP'] + res['FN'])
    res['F1-score'] = 2 / (1 / res['Precision'] + 1 / res['Recall'])
    res['False Positive Rate'] = res['FP'].div(res['FP'] + res['TN'])

    res['% of Total'] = res['Total'].div(res['Total'].sum()) * 100
    res['% of Positives'] = res['Positives'].div(res['Positives'].sum()) * 100

    res = res[['Mean Score',
               'Total',
               'Positives',
               'Positive Rate',
               '% of Total',
               '% of Positives',
               'Accuracy',
               'Precision',
               'Recall',
               'False Positive Rate']]
    res.index += 1

    return res


def plot_roc(y, predictions):
    """
	Gráfica de la curva de roc del modelo

	Args:
		y (array): Vector de variable objetivo
		predictions (array): Vector de predicciones
	"""
    fpr, tpr, thresholds = roc_curve(y,
                                     predictions,
                                     pos_label=None,
                                     sample_weight=None,
                                     drop_intermediate=True)
    plt.subplot(121)
    plt.plot(fpr, tpr)
    auc = np.trapz(tpr, fpr).round(5)
    plt.axis([0, 1, 0, 1])
    plt.plot([0, 1], [0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.text(0.65, 0.02, 'AUC: ' + str(auc), fontsize=12)
    plt.title('ROC Curve')


def plot_prc(y, predictions):
    """
	Gráfica de la curva de precision-recall del modelo

	Args:
		y (array): Vector de variable objetivo
		predictions (array): Vector de predicciones
	"""
    average_precision = average_precision_score(y, predictions).round(5)
    precision, recall, threshold = precision_recall_curve(y, predictions)
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})

    plt.subplot(122)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.text(0.65, 0.02, 'AP: ' + str(average_precision), fontsize=12)
    plt.title('Precision-Recall Curve')


def show_metrics(y, predictions, sc=0.5, disp=True, n=10):
    """
	Muestra precisión por cuantiles, precisión por intervalos
	de score y gráficas de PRC y ROC

	Args:
		y (array): Instancias de la variable dependiente
		predictions (array): Predicciones
		sc (float): Score de corte entre 0 y 1 que marca el límite de clasificación
					(arriba de sc se considera positivo)
		disp (boolean): Imprimir matriz con métricas
		n (int): Número de cuantiles
	"""
    yt = y.copy()
    predictionst = predictions.copy()
    metrics = model_precision(yt, predictionst, sc, disp)
    plot_roc(yt, predictionst)
    plot_prc(yt, predictionst)
    plt.subplots_adjust(left=3.1, right=5.1, bottom=2, top=3, hspace=0.2,
                        wspace=0.5)
    plt.show()
