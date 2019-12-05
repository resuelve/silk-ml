import random
import logging
from .entrena import entrena_red


class Red():
    """
    Clase que representa una red neuronal simple  de varias capas (MLP).
    """

    def __init__(self, nn_param_candidatos=None):
        """Inicializa nuestra red.

        Args:
            nn_param_candidatos (dict): parámetros que puede incluir la red.
            Por ejemplo:
                num_neurons (list): [64, 128, 256]
                num_capas (list): [1, 2, 3, 4]
                activacion (list): ['relu', 'elu']
                optimizador (list): ['rmsprop', 'adam']
        """
        self.accuracy = 0.  # aún no tiene presición, aún no se entrena
        self.nn_param_candidatos = nn_param_candidatos
        self.red = {}  # (dic): representa la red recien instanciada

    def red_aleatoria(self):
        """Crea una red aleatoria a partir de los parámetos provistos"""
        for key in self.nn_param_candidatos:
            self.red[key] = random.choice(self.nn_param_candidatos[key])

    def create_red(self, red):
        """Asigna las propiedades de una red.

        Args:
            red (dict): la representación de una red en diccionario

        """
        self.red = red

    def entrena(self, datos_listos):
        """Entrena la red y guarda su precisión.

        Args:
            datos_listos(tuple): tupla con los datos que van a utilizarse

        """
        self.accuracy = entrena_red(self.red, datos_listos)

    def imprime_red(self):
        """Imprime la representación de la red y su precisión."""
        logging.info(self.red)
        logging.info("Accuracy de la red: %.2f%%", self.accuracy * 100)
