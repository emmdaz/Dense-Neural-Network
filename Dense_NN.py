
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import numpy as np

#Se carga la base de datos MNIST

ds = mnist.load_data()

# Separación en datos de entrenamiento y datos de prueba

(x_train,y_train),(x_test,y_test)=ds
# print(x_train.shape,x_test.shape)

# Se hace al aplanado de las imágenes
x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(10000,784)

# Se modifica el tipo de datos a tipo flotante (para evitar conflictos)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# Se normalizan los datos acorde a la intensidad de los píxeles
x_train /= 255.
x_test /= 255.

# Se realiza el one-hot enconding:

# Se definen el número de clases que hay en la base de datos
classes = 10

# Se realiza la conversión:
y_train = keras.utils.to_categorical(y_train,classes)
y_test = keras.utils.to_categorical(y_test,classes)

# Se configuran algunos aspectos del modelo:

# Número de neuronas de la capa oculta
n = 15

# Valor del Learning Rate
eta = 3.0
eta = float(eta) # Tiene que ser un dato de tipo flotante, pues de lo contrario
                 # tiene conflicto

# Número de épocas
epochs = 30

# Tamaño del mini-batch
mini_batch=10

# Se realiza la creación del modelo a como se había definido usando Numpy:
model = Sequential([
    Dense(n, activation='sigmoid', input_shape=(784,)),
    Dense(classes, activation='sigmoid')
])
# El modelo es secuencial, se van uniendo neurona por neurona de forma 
# consecutiva

model.compile(loss='binary_crossentropy',optimizer=SGD(learning_rate=eta),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size = mini_batch,
                    epochs = epochs,
                    verbose = 1,
                    validation_data = (x_test, y_test)
                    )

mod = model.predict(x_test)

print("Resultado predicho: ")
print(np.where(mod[6] == np.max(mod[6])))
print("Resultado correcto: ")
print(np.where(y_test[6] == np.max(y_test[6])))



