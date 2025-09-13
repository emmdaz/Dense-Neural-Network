
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
import numpy as np

import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
wandb.require("core")
wandb.login()

###############################################################################

wandb.init(
    project="Experiments Series 1.01",
    config={
        "Layer1": 256,
        "Activation_1": "relu",
        "Dropout1": "No dropout",
        "Layer2": 256,
        "Activation_2": "sigmoid",
        "Dropout2": "No dropout",
        "Layer3": 10,
        "Activation_3": "softmax",
        "Dropout3": "No dropout",
        "Optimizer": "adam",
        "Metric": "accuracy",
        "Epoch": 30,
        "Batch_size": 10,
        "Eta": 1e-5,
        "L2": 1e-5,
        "Loss": "binary_crossentropy"
    }
)

config = wandb.config

###############################################################################

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
classes = config.Layer3

# Se realiza la conversión:
y_train = keras.utils.to_categorical(y_train,classes)
y_test = keras.utils.to_categorical(y_test,classes)

# Se configuran los hiperparámetros del modelo:

# Valor del Learning Rate
eta = config.Eta
eta = float(eta) # Tiene que ser un dato de tipo flotante, pues de lo contrario
                 # tiene conflicto

# Número de épocas
epochs = config.Epoch

# Tamaño del mini-batch
mini_batch= config.Batch_size

# Se realiza la creación del modelo a como se había definido usando Numpy:
model = Sequential([
    Dense(config.Layer1, activation = config.Activation_1, input_shape = (784,),
          kernel_regularizer = regularizers.L2(config.L2)),
    Dense(config.Layer2, activation = config.Activation_2,
          kernel_regularizer = regularizers.L2(config.L2)),
    Dense(classes, activation = config.Activation_3,
          kernel_regularizer = regularizers.L2(config.L2))
])
model.summary()
# El modelo es secuencial, se van uniendo neurona por neurona de forma consecutiva

###############################################################################

model.compile(loss=config.Loss, optimizer = config.Optimizer,
              metrics=[config.Metric])

history = model.fit(x_train, y_train,
                    batch_size = mini_batch,
                    epochs = epochs,
                    verbose = 1,
                    validation_data = (x_test, y_test),
                    callbacks=[WandbMetricsLogger(log_freq=5),
                               WandbModelCheckpoint("models.keras")]
                    )
wandb.finish()

mod = model.predict(x_test)

print("Resultado predicho: ")
print(np.where(mod[6] == np.max(mod[6])))
print("Resultado correcto: ")
print(np.where(y_test[6] == np.max(y_test[6])))



