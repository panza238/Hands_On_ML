# superclassing a model. Una manera más programadora de programar!
import tensorflow as tf
import tensorflow.keras as keras

class WideAndDeepModel(keras.Model):
    """Esta clase va a ser mi modelo..."""

    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs) # handles standard args (e.g., name).
        # El super() acá hacer referencia a la clase madre, la clase de la que se está heredando. En este caso, es la clase keras.Model.
        self.hidden1 = keras.layers.Dense(units, activation=activation) # Creo mis dos layers
        self.hidden2 = keras.layers.Dense(units, activation=activation)
        self.main_output = keras.layers.Dense(1)  # Tengo 2 outputs: una main y la otra aux. 
        self.aux_output = keras.layers.Dense(1)

    def call(self, inputs):
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        # hasta acá, tengo input_B --> hidden_1 --> hidden_2
        concat = keras.layers.concatenate([input_A, hidden2])
        # ahora, con el concatenate, conecto input_A --> concat y hidden_2 --> concat
        # CONCAT ES UNA LAYER EN SÍ!
        main_output = self.main_output(concat)  # Esta output sale directamente del concat
        aux_output = self.aux_output(hidden2)  # Esta sale directamente de hidden_2
        # Es una acquitectura más complicada, y debe tener sus usos.
        return main_output, aux_output

# Instancio la clase que acabo de crear.
model = WideAndDeepModel()
# Al instanciarla, se definen todas las layers. Cuando llamo al método .call(), ejecuto todo como si fuera la Functional API


print("\nTEST OK!\n")
