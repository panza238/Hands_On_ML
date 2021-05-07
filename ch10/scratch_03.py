# WIDE AND DEEP NEURAL NETWORKS USING FUNCTIONAL API.
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
input_ = keras.layers.Input(shape=X_train.shape[1:])  # input layer. Le digo la shape que tiene que tener
hidden_1 = keras.layers.Dense(30, activation="relu")(input_)
# Este approach es fundamentalmente diferente. NO es OOP. Es funcional! Notar que, apenas creamos la Dense, la usamos como función (el parámetro que le pasamos a la función sería la previous layer)
hidden_2 = keras.layers.Dense(30, activation="relu")(hidden_1)
# Lo mimso que con la capa anterior: La layer es una función. Su parámetro de entrada es hidden_1 (la previous layer).
# Entonces: la ANN no se crea como objeto. Se crea como concatencación de función, como un procedimiento. No es OOP. ES FUNCIONAL.
# Hasta acá, la parte Deep de la ANN. Una secuencia de layers como las que veníamos creando antes. 

concat = keras.layers.Concatenate()([input_, hidden_2])  # Este concat es la magia de la parte WIDE de la network. 
# Haciendo esto, se "puentea" la hidden_1. Se pasa directamente de la input_ a la hidden 2. Es la parte WIDE. 
# NOTAR QUE CONCAT ES UNA LAYER NUEVA. Creo la función y le paso la lista de layers a seguir. 

output_ = keras.layers.Dense(1)(concat)
# output se crea como otra función. Y se conecta a la concat. Entonces quedan 2 "caminos"
# input_ --> hidden_1 --> hidden_2 --> concat --> output 
# input_ --> hidden_2 --> concat --> output

model = keras.Model(inputs=[input_], outputs=[output_])


# Same as before: compile, fit, evaluate...
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
