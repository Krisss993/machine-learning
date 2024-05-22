import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LinearRegression:
    def __init__(self, eta=0.01, epochs=100, is_verbose=True):
        self.eta=eta
        self.epochs=epochs
        self.is_verbose=is_verbose
        self.list_of_errors = []
        
    def predict(self, X):
        ones = np.ones((X.shape[0],1))
        X_1 = np.append(X.copy(), ones, axis=1)
        return self.get_activation(X_1)
    
    
    def get_activation(self, x):
        activation = np.dot(x, self.w)
        return activation
    
    
    def fit(self, X, y):
        
        ones = np.ones((X.shape[0],1))
        X_1 = np.append(X.copy(), ones, axis=1)
        
        self.w = np.random.rand(X_1.shape[1])
        
        for e in range(self.epochs):
            
            error = 0
            
            activation = self.get_activation(X_1)
            delta_w = self.eta * np.dot((y-activation), X_1)
            self.w+=delta_w
            
            error = np.square(y - activation).sum()/2.0
           
            self.list_of_errors.append(error)
            
            if (self.is_verbose):
                print('Epoch: {}, weights: {}, errors {}'.format(e, self.w, error))








wine = pd.read_csv(r'F:/\UdemyMachineLearning/winequality_/winequality-red.csv', sep=';')
wine
wine.info()

X = wine['alcohol'].values.reshape(-1,1)
y = wine['quality'].values

scaler = StandardScaler()
scaler.fit(X)
X_std = scaler.transform(X)



X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2)

lr = LinearRegression(eta=0.0001, epochs=100, is_verbose=True)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

plt.scatter(range(lr.epochs), lr.list_of_errors)
plt.scatter(y_test, y_pred, s=80, facecolors = 'none', edgecolors='r')


plt.figure(figsize=(7,7))
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color='red')

# OCENA MODELU %
np.count_nonzero(np.rint(y_pred) == y_test) / len(y_test)


round(np.mean(y_test))
# OCENA LOSOWA %
np.count_nonzero(round(np.mean(y_test)) == y_test) / len(y_test)







