# WIDE AND DEEP NEURAL NETWORKS USING FUNCTIONAL API. MULTIPLE INPUTS!
import tensorflow as tf
import tensorflow.keras as keras

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing_df = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing_df.data, housing_df.target
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full
)

scaler = StandardScaler()  # scaler es una instance de la clase StandardScaler
X_train = scaler.fit_transform(X_train)  # (Hago el fit y el transform en un solo paso)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Creo mi wide and deep nn. 
print(X_train)
print(X_train.shape)
print(X_train.shape[1])
print(X_train.shape[1:])  # X_train.shape[0] = instancias (filas), X_train.shape[1] = features (columnas)
# usa [1:] para cubrirse en caso de que el array de input tenga más de 2 dimensiones. 

# Creo mi wide and deep NN
input_A = keras.layers.Input(shape=[5], name="wide_input")
input_B = keras.layers.Input(shape=[6], name="deep_input")
# Acá acabo de crear dos inputs! Mando 5 features por el camino WIDE (0 a 4), y 6 inputs por el camino DEEP (2 a 7). Las features se solapan!
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1, name="output")(concat)
model = keras.Model(inputs=[input_A, input_B], outputs=[output])  # Notar que se definió una lista de inputs: [input_A, input_B]


# NOT THE SAME AS BEFORE!  SEE DIFFERENT STEPS AT COMPILING AND FITTING! 
# LE TENGO QUE DECIR QUÉ FEATURES VAN EN CADA INPUT. 

model.compile(
    optimizer="sgd",
    loss="mean_squared_error"
)

history = model.fit(
    x=X_train, y=y_train,
    validation_data=(X_valid, y_valid)
)

model.evaluate(x=X_test, y=y_test)

# predict
X_new = X_test[:5]
predictions = model.predict(X_new)
print("\n")
print(X_new)
print(predictions)
