from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

# si después de 5 epochs no mejora se detiene el entrenamiento de ese modelo
EARLY_STOPPER = EarlyStopping(patience=5)


def compila_modelo(red, outputs, inputs):
    """Compila un modelo secuencial.

    Args:
        red (dict): diccionario que contiene los parámetros de la red
        outputs(int): número de entradas de la red
        inputs(int): número de salida de la red
    Returns:
        model(Keras.model): Una red compilada.
    """
    # Obtenemos los parámetros de nuestra red.
    num_capas = red['num_capas']
    num_neurons = red['num_neurons']
    activacion = red['activacion']
    optimizador = red['optimizador']

    model = Sequential()

    # Se añade cada capa.
    for i in range(num_capas):
        # Necesitamos el número de inputs para la primer capa.
        if i == 0:
            model.add(
                Dense(num_neurons, activation=activacion, input_shape=inputs))
        else:
            model.add(Dense(num_neurons, activation=activacion))

        model.add(Dropout(0.5))
        # le añadimos una capa de Dropout antes de la última para mejor desempeño
        # Hinton (2012)
    # Capa de salida.
    model.add(Dense(outputs, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizador,
                  metrics=['accuracy'])

    return model


def entrena_red(red, datos_listos):
    """Entrena el modelo, regresa su evaluacion.

    Args:
        red (dict): los parámetros de una red
        outputs(int): número de outputs que queremos que nuestra red tenga
        datos_listos(tuple): tupla con los datos que van a utilizarse:
            x_train(np.array): valores de variables de entrenamiento
            x_test(np.array): valores variables de prueba
            y_train(np.array): valores de variable a predecir de entrenamiento
            y_test(np.array): valores de variable a predecir de prueba
    Returns:
        res_accuracy(float): accuracy del modelo
    """
    x_train, x_test, y_train, y_test = datos_listos
    # número de inputs de la red a partir de # de datos
    inputs = (x_train[0].size,)
    outputs = 1
    model = compila_modelo(red, outputs, inputs)

    model.fit(x_train, y_train,
              batch_size=128,  # esto también puede ser un parámetro a optimizar
              epochs=10000,  # usamos EarlyStopping así que no es el límite real
              verbose=0,
              validation_data=(x_test, y_test),
              callbacks=[EARLY_STOPPER])

    score = model.evaluate(x_test, y_test, verbose=0)
    res_accuracy = score[1]  # 1 es accuracy. 0 es loss.

    return res_accuracy
