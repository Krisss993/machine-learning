import numpy as np
import tensorflow
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Convolution2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from keras.models import load_model



(x_train,y_train), (x_test,y_test) = cifar10.load_data()
x_train.shape

x_train = (x_train / 255.0) - 0.5
x_test = (x_test / 255.0) - 0.5
print(x_train.min(), "-", x_train.max())

model = Sequential([
   Convolution2D(filters=64, kernel_size=(3,3), input_shape=(32,32,3), activation='relu', padding='same'),
   Convolution2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'),
   MaxPool2D((2,2)),
   Convolution2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'),
   Convolution2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'),
   MaxPool2D((2,2)),
   Convolution2D(filters=16, kernel_size=(3,3), activation='relu', padding='same'),
   Convolution2D(filters=16, kernel_size=(3,3), activation='relu', padding='same'),
   Flatten(),
   Dense(units=512, activation="relu"),
   Dense(units=64, activation="relu"),
   Dense(units=10, activation="softmax")
])

model.summary()


optim = SGD(learning_rate=0.001, momentum=0.5)
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
   x_train,
   to_categorical(y_train),
   epochs=80,
   validation_split=0.15,
   verbose=1
)

model.save(r'./logisticRegressionKeras.hdf5')

# Wczytaj model
# loaded_model = load_model('logisticRegressionKeras.hdf5')

eval = model.evaluate(x_test, to_categorical(y_test))
eval


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
