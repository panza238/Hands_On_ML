# First dive into tf and keras
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import numpy as np

print(tf.__version__)
print(keras.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
print(X_train_full.shape)  # OK. Esto devuelve (60000, 28, 28) ==> 60000 imágenes de 28 pixels por 28 pixels.
print(X_train_full.dtype)  # 8 bit unsigned integer 
print(y_test.shape)
print(y_test.dtype)

# Por default, el dataset MNIST (que está en la DB de keras) viene separado en train y test
# Como no viene con un dataset de validación, lo creo yo. Me dejo un 10% para validación.
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# Defino los nombres de las 10 clases (EN ORDEN). Me aprece que esto se podría haber hecho mejor con un diccionario...

model = keras.models.Sequential()  ## Instancio la clase Sequential para crear un modelo. 
model.add(keras.layers.Flatten(input_shape=[28, 28]))  ## Agrego mi primera layer: esta layer aplana la entrada. las imágenes pasan de ser un array 2D de 28x28 y pasan a ser un array 1D de 784
# Si quiero pensar en términos de OOP: .add() es un método que modifica (o afecta) al objeto model, que es una instancia de la clase Sequential.
# Flatten es como ejejcutar X.reshape(-1, 1). Transforma todo a un array 1D
model.add(keras.layers.Dense(300, activation="relu"))  ## Acá creo la primera capad de mi ANN.
# Recordar que una "Dense" es una layer cuyos outputs están todos conectados a los inputs de la siguiente layer. 
# Repaso de terminología. 
# Perceptron --> Layer de TLUs. TLU (threshold logic unit)--> n inputs, 1 output ("neurona"). Dense Layer --> TODOS los outputs de la layer anterior, conectados a TODOS los inputs de la layer actual.
# La función de activación de esto es la "relu" rectified linear unit. (lineal del para valores positivos. 0 para valores negativos)
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))  ## ATENTI. No es casualidad que la última capa tenga 10 unidades ("neuronas"), 10 TLUs.
# Tiene una TLU por clase. Cada TLU va a devolver la probabildad de que la imagen pertenezca a una clase en particular.
# Uso softmax porque las clases son mutuamente excluyentes.

# AHORA VIENE LA PARTE DE PEDIRLE A KERAS QUE ME DE INFO DEL MODELO.
# RESUMEN GENERAL DEL MODELO.
model.summary() # model.summary() me muestra información sobre cada una de las layers. 
# GETTING SPECIFIC LAYERS:
hidden_1 = model.layers[1]
print("Layer name:\t", hidden_1.name)  # esto me va a devolver "dense", el nombre de la segunda layer
print(hidden_1)
print(".get_layer() method\t", model.get_layer("dense") is hidden_1)  # con get_layer, literalmente tomo una layer.
# GETTING WIGHTS
weights, biases = hidden_1.get_weights()
print("dense layer weights:\n", weights)
print("dense layer biases:\n", biases) 
# Notar que todos los weights se inicializan en un valor random, y que todos los biases (término independiente), se inicializan en 0.
print("weights shape", weights.shape)
print("biases shape", biases.shape)  # Notar que .shape no es un method. Es un attribute del objeto. 
# Es buena práctica decirle la input_shape al modelo, pero no es estrictamente necesario. Lo puede inferir al momento del entrenamiento. 

# COMPILANDO EL MODELO:
model.compile(
    loss=keras.metrics.sparse_categorical_crossentropy,
    optimizer=keras.optimizers.SGD(),
    metrics=[keras.metrics.sparse_categorical_accuracy]
)  # Compilo el modelo. Defino la loss function, el método de optimización, y qué métrica quiero usar. 
# lo de sparse categorical es porque tengo labels definidos (un número del 0 al 9). Si esto estuviera encoded, por ejemplo, (cada target es un vector con todo 0 y un 1 en la posición correspondiente a ese target), no pondría sparse. Sería únicamente categorical.

# FITTEANDO EL MODELO (igual que sci-kit)
history = model.fit(
    x=X_train, y=y_train, epochs=30,
    validation_data=(X_valid, y_valid)
)
# OJO! Algo muy piola es que se le puede dar distintos weights a las clases, si nuestro dataset está bised.
# Pero también se le puede dar distintos pesos a cada registro (si le tengo más fe a algún registro que a otro).
# el método .fit() devuelve un history object. (ahora tengo dos objetos: el modelo, y el history). 
print("History params:", history.params)
print("History values", history.history)  # Esto me devuelve un dictionary cuyos keys son las métricas que definí, y sus values son listas con los valores correspondientes a cada epoch.

# Gráfico de la evolución del algoritmo:
history_df = pd.DataFrame(history.history)
history_df.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # gca --> get current axes. method que devuelve los axes del objeto plot.
plt.show()

# Evaluación del modelo:
model.evaluate(X_test, y_test)

# Predicciones:
X_new = X_test[-5:]
y_prob = model.predict(X_new)
print("\nProbabilities", y_prob.round(3))  # y_prob es la probabilidad. 

y_pred = model.predict_classes(X_new)  # y_pred es la predicción de la clase (la clase con mayor predicted prob)
# Como, aparentemente, predict_classes está deprecado, la otra opción es hacer esto:
np.argmax(model.predict(x), axis=-1)
print("\nPredictions", y_pred)

