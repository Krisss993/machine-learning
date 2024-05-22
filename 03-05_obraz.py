import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train.shape
x_train.shape

il_zdjec = 16
zdjecia = np.zeros((il_zdjec,32,32,3), dtype=int)
opisy = np.zeros((il_zdjec,1), dtype=int)
for i in range(il_zdjec):
   indeks = np.random.randint(0, 50000)
   zdjecia[i] = x_train[indeks]
   opisy[i] = y_train[indeks]
   
slownik = {
   0: 'samolot',
   1: 'samochód',
   2: 'ptak',
   3: 'kot',
   4: 'jeleń',
   5: 'pies',
   6: 'żaba',
   7: 'koń',
   8: 'statek',
   9: 'ciężarówka',
}

fig = plt.figure()
for n, (obrazek, label) in enumerate(zip(zdjecia, opisy)):
   a = fig.add_subplot(4, 4, n + 1)
   plt.imshow(obrazek)
   a.set_title(slownik[label[0]])
   a.axis('off')
fig.set_size_inches(fig.get_size_inches() * il_zdjec / 7)
plt.show()

x_train = x_train.reshape((-1, 3072))
x_test = x_test.reshape((-1, 3072))
x_train.shape

x_train = (x_train / 255)- 0.5
x_test = (x_test / 255)- 0.5

model = Sequential([
   Dense(1024, activation='tanh', input_shape=(3072,)),
   Dense(512, activation='tanh'),
   Dense(256, activation='tanh'),
   Dense(128, activation='tanh'),
   Dense(64, activation='tanh'),
   Dense(10, activation='softmax')
])

model.compile(
   optimizer='RMSprop',
   loss='categorical_crossentropy',
   metrics=['accuracy']
)

model.fit(
   x=x_train,
   y=to_categorical(y_train),
   epochs=15,
   shuffle=True
)

