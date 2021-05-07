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

# De nuevo, uso la sequential API para armar mi modelo:
model = keras.models.Sequential([
    keras.layers.Dense(units=30, activation="relu", input_shape=X_train.shape[1:]),  # No entiendo por qué va el [1:]
    keras.layers.Dense(1)
])
# Tengo una sola unit en la Dense de salida, porque solo quiero predecir el precio! 1-D output!

# compilo:
model.compile(
    optimizer="sgd",
    loss="mean_squared_error"
)

# fitteo:
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid))

# testeo:
mse_test = model.evaluate(X_test, y_test)

# predict
X_new = X_test[:3]  # "obervaciones nuevas"
y_pred = model.predict(X_new)
print("prediction", y_pred)