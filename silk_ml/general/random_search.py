import random
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_curve
from silk_ml.evaluation.metrics import model_precision


def NN(X_train, y_train, neurons, activations, initializer,
       optimizer, epochs, batch, loss, checkpoint=False):
    """
    Entrenamiento de una red neuronal

    Args:
        X_train (Array): Variables independientes (muestra de entrenamiento)
        y_train (Array): Variable objetivo (muestra de entrenamiento)
        neurons (list): Número de neuronas en cada capa
        activations (list): Función de activación en cada capa
        initializer (str): Kernel initializer
        optimizer (str): Optimizer
        epochs (int): Número de epochs
        batch (int): Tamaño de cada batch
        loss (str): Función de pérdida
        checkpoint (boolean): Guarda modelo en cada mejora
    Returns:
        model (modelo): Modelo de Red Neuronal
    """
    dim = X_train.shape[1]
    model = Sequential()
    model.add(Dense(neurons[0],
                    input_dim=dim,
                    kernel_initializer=initializer,
                    bias_initializer='zeros',
                    activation=activations[0]))
    for i in range(1, len(neurons)):
        model.add(Dense(neurons[i],
                        kernel_initializer=initializer,
                        bias_initializer='zeros',
                        activation=activations[i]))
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    if checkpoint != False:
        filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath,
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='max')
        callbacks_list = [checkpoint]
        model.fit(X_train,
                  y_train,
                  epochs=epochs,
                  batch_size=batch,
                  callbacks=callbacks_list,
                  validation_split=0.1)
    else:
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch)

    return model


def random_nets(X_train, y_train, it, epcs):
    """
    Creación de redes con parámetros aleatorios

    Args:
        X_train (Array): Variables independientes (muestra de entrenamiento)
        y_train (Array): Variable objetivo (muestra de entrenamiento)
        it (int): Número de iteraciones (redes) que queremos entrenar
        epcs (int): Número de epochs para cada mini batch en todas las redes
    Returns:
        modelos (list): Lista de redes neuronales
    """
    activation_functions = ['relu', 'sigmoid', 'tanh',
                            'elu', 'selu', 'softmax',
                            'softplus', 'softsign',
                            'hard_sigmoid', 'linear']
    initializers = ['he_normal', 'he_uniform',
                    'glorot_normal', 'glorot_uniform',
                    'lecun_normal']
    optimizers = ['Adam', 'SGD', 'RMSprop',
                  'Adagrad', 'Adadelta', 'Adamax',
                  'Nadam']
    batches = [64, 128, 256, 512, 1024, 2048]
    modelos = []
    parameters = pd.DataFrame()
    for i in range(it):
        k = round(abs(np.random.randn() * 10))
        if k != 0:
            try:
                neurons = [X_train.shape[1]]
                neurons2 = [round(abs(np.random.randn() * 100))
                            for j in range(k)]
                neurons.extend(neurons2)
                neurons.append(1)
                activations = [random.choice(activation_functions)
                               for j in range(k+1)]
                activations.append('sigmoid')
                initializer = random.choice(initializers)
                optimizer = random.choice(optimizers)
                batch = random.choice(batches)
                loss = 'binary_crossentropy'
                rows = np.zeros((len(neurons), 6))
                params = pd.DataFrame(rows, columns=['Neurons',
                                                     'Activation',
                                                     'Batch Size',
                                                     'Initializer',
                                                     'Optimizer',
                                                     'Loss'])
                params['Neurons'] = neurons
                params['Activation'] = activations
                params['Batch Size'] = batch
                params['Initializer'] = initializer
                params['Optimizer'] = optimizer
                params['Loss'] = loss
                modelos.append(NN(X_train,
                                  y_train,
                                  neurons,
                                  activations,
                                  initializer,
                                  optimizer,
                                  epochs=epcs,
                                  batch=batch,
                                  loss=loss))
            except Exception as e:
                logging.error(e)

    return modelos


def random_search(X_test, y_test, modelos, metric, sc=0.5):
    """
    Random search en parámetros de una red neuronal de clasificación

    Args:
        modelos (list): Lista de redes neuronales
        X_test (array): Variables independientes (muestra de prueba)
        metric (str): Métrica de precisión para seleccionar mejor modelo
                      ['acc', 'prec', 'rec', 'f1', 'mcc', 'auc']
        sc (float): Threshold para métricas en (0,1)
    Returns:
        best_model (modelo): Mejor red seleccionada

    """
    plt.figure(figsize=(12, 12))
    max_met = 0
    best_model = modelos[0]
    for i in modelos:

        predictions_test = i.predict(X_test)
        acc, prec, rec, f1, mcc = model_precision(y_test,
                                                  predictions_test,
                                                  sc,
                                                  disp=False)
        fpr, tpr, thresholds = roc_curve(y_test,
                                         predictions_test,
                                         pos_label=None,
                                         sample_weight=None,
                                         drop_intermediate=True)
        plt.plot(fpr, tpr)
        auc = np.trapz(tpr, fpr)
        dic = {'acc': acc,
               'prec': prec,
               'rec': rec,
               'f1': f1,
               'mcc': mcc,
               'auc': auc}
        met = dic[metric]
        if met > max_met:
            max_met = met
            best_model = i

    plt.axis([0, 1, 0, 1])
    plt.plot([0, 1], [0, 1])
    plt.text(0.65, 0.02, metric + ': ' + str(max_met), fontsize=12)
    plt.show()

    return best_model


def best_nn(X_train, y_train, X_test, y_test, it, epcs, metric, threshold=0.5):
    """
    Selección de mejor modelo de red neuronal

    Args:
        X_train (Array): Variables independientes (muestra de entrenamiento)
        y_train (Array): Variable objetivo (muestra de entrenamiento)
        X_test (Array): Variables independientes (muestra de prueba)
        y_ttest (Array): Variable objetivo (muestra de prueba)
        it (int): Número de iteraciones (redes)
        epcs (int): Número de epochs para cada mini batch en todas las redes
        metric (str): Métrica de precisión para seleccionar mejor modelo
                      ['acc', 'prec', 'rec', 'f1', 'mcc', 'auc']
        threshold (float): threshold para métricas en (0,1)
    Returns:
        bm (modelo): Mejor red seleccionada
    """
    modelos = random_nets(X_train, y_train, it=it, epcs=epcs)
    bm = random_search(X_test, y_test, modelos, metric=metric, sc=threshold)

    return bm
