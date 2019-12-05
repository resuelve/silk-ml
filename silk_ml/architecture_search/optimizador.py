import random
from operator import add
from functools import reduce
from .red import Red


class Optimizador():
    """
        Clase que implementa los algoritmos genéticos para optimizar las redes candidatas.
        """

    def __init__(self, nn_params, retiene=0.4,
                 random_selec=0.1, proba_muta=0.2):
        """ Crea un optimizador.
                Args:
                        nn_params (dict): partes de una red a crear
                        retiene (float): porcentaje de la población que se retiene después
                                                        de cada generación
                        random_selec (float): probabilidad de que una red rechazada se quede
                                                                        en la población
                        proba_muta (float): Probabilidad de que una red se mute aleatoriamente
                """
        self.proba_muta = proba_muta
        self.random_selec = random_selec
        self.retiene = retiene
        self.nn_params = nn_params

    def crea_poblacion(self, tam):
        """Crea una población de redes aleatorias.
                Args:
                        tam (int): tamaño de la población
                Returns:
                        poblacion (list): lista de objetos Red
                """
        poblacion = []
        for _ in range(0, tam):
            # Crea una red neuronal aleatoria.
            red = Red(self.nn_params)
            red.red_aleatoria()

            # se añade la red a la población.
            poblacion.append(red)

        return poblacion

    @staticmethod  # para utilizarlo desde una instancia de la clase
    def aptitud(red):
        """
                Devuelve la precisión. Que es nuestra variable de aptitud
                """
        return red.accuracy

    def avg_pob(self, poblacion):
        """Encuentra el promedio de aptitud de una población.
                Args:
                        poblacion (list): la poblacion de redes
                Returns:
                        avg (float): el promedio del aptitud de la población

                """
        suma = reduce(add, (self.aptitud(red) for red in poblacion))
        avg = suma / float((len(poblacion)))
        return avg

    def cruza(self, madre, padre):
        """Hace dos redes hijas a partir de sus padres.
                Args:
                        madre (dict): parámetros de una red
                        padre (dict): parámetros de una red
                Returns:
                        hijos(list): Una lista con dos hijos nuevos
                """
        hijos = []
        for _ in range(2):

            hijo = {}

            # Se recorren los parámetros de las redes
            # y se elige aleatoriamente el valor del hijo a partir de sus padres
            for param in self.nn_params:
                hijo[param] = random.choice(
                    [madre.red[param], padre.red[param]]
                )

            # Se crea un objeto Red nuevo
            red = Red(self.nn_params)
            red.create_red(hijo)

            # Se eligen aleatoriamente algunos hijos para mutarlos
            if self.proba_muta > random.random():
                red = self.muta(red)

            hijos.append(red)

        return hijos

    def muta(self, red):
        """ Aleatoriamente muta una parte de la red.
                Args:
                        red (dict): los parámetros de una red a mutar
                Returns:
                        (Network): una red mutada aleatoriamente
                """
        # Se elige una característica aleatoria
        mutacion = random.choice(list(self.nn_params.keys()))

        # Muta uno de los parámetros
        red.red[mutacion] = random.choice(self.nn_params[mutacion])

        return red

    def evoluciona(self, poblacion):
        """ Evoluciona una población de redes.
                Args:
                        poblacion (list): Una lista de parámetros de redes
                Returns:
                        padres (list): La lista de parámetros de redes evolucionada
                """
        # Obtenemos el accuracy de cada red
        evaluacion = [(self.aptitud(red), red) for red in poblacion]

        # Ordenamos por accuracy
        evaluacion = [x[1] for x in sorted(
            evaluacion, key=lambda x: x[0], reverse=True)]

        # Obtenemos el número de redes definido para cada iteración
        retiene_len = int(len(evaluacion) * self.retiene)

        # lospadres son todas las redes más aptas que queremos retener
        padres = evaluacion[:retiene_len]

        # retenemos pocos de los menos aptos. Aleatoriamente
        for individuo in evaluacion[retiene_len:]:
            if self.random_selec > random.random():
                padres.append(individuo)

        # Vemos cuantos hijos debemos obtener
        padres_len = len(padres)
        tam_deseado = len(poblacion) - padres_len
        hijos = []

        # Se añaden hijos por cada variedad creada por pareja de padres.
        while len(hijos) < tam_deseado:

            # hacemos parejas aleatoria de padres.
            padre = random.randint(0, padres_len - 1)
            madre = random.randint(0, padres_len - 1)

            # probamos que no sean la misma red seleccionada madre y padre
            if padre != madre:
                padre = padres[padre]
                madre = padres[madre]

                # Hacen hijos.
                bebes = self.cruza(padre, madre)

                # se añaden los hijos uno a la vez.
                for bebe in bebes:
                    # No se añaden más hijos que el tamaño deseado
                    if len(hijos) < tam_deseado:
                        hijos.append(bebe)

        padres.extend(hijos)
        return padres
