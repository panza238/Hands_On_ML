# reegression problem with california housing prices dataset.
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# get data
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
# split data
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)
# scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Model:
model = keras.models.Sequential([
        keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
        keras.layers.Dense(1)
])
# compile
model.compile(loss="mean_squared_error", optimizer="sgd")
history = model.fit(X_train, y_train, epochs=20,
            validation_data=(X_valid, y_valid))
# evaluate
mse_test = model.evaluate(X_test, y_test)

# Lo mismo, se puede hacer model.predict(<observation>) y se predice
 