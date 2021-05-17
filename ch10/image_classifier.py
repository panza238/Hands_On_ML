# Building a simple image classifier with the Sequential API
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
# Separo en training y validation dataset:
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
# el validation me va a servir para saber si estoy overfitteando mientras entreno. El test es la prueba final. 
# image class names. 
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Model:
model = keras.models.Sequential()  #instancio en modelo.
# agrego capas al modelo:
model.add(keras.layers.Flatten(input_shape=[28, 28])) 
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
# Compile:
model.compile(loss="sparse_categorical_crossentropy",
                optimizer="sgd",
                metrics=["accuracy"])
# Fit:
history = model.fit(X_train, y_train, epochs=30,
            validation_data=(X_valid, y_valid))

# Plot accuracy and loss:
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()

# Final evaluation:
model.evaluate(X_test, y_test)

# A partir de esto, se pueden hacer predcciones como en scikit-learn: model.predict()

print("\nTEST OK!\n")