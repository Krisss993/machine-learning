import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, eta=0.1, epochs=50, is_verbose=False):
        self.eta=eta
        self.epochs=epochs
        self.is_verbose=is_verbose
        
    def __predict(self, x):
        total_prediction = np.dot(x, self.w)
        y_pred = np.where(total_prediction>0,1,-1)
        return y_pred
    
    def predict(self, X):
        ones = np.ones((X.shape[0],1))
        X_1= np.append(X.copy(), ones, axis=1)
        return self.__predict(X_1)
    
    def fit(self, X,y):
        ones = np.ones((X.shape[0],1))
        X_1= np.append(X.copy(), ones, axis=1)
        self.w = np.array([0.38544573,0.40046814,0.50987714,0.94058191])
        self.list_of_errors=[]
        for e in range(self.epochs):
            nr_of_errors=0
            y_pred = self.__predict(X_1)
            delta_w = self.eta * np.dot((y-y_pred),X_1)
            self.w+=delta_w
            nr_of_errors = np.count_nonzero(y-y_pred)
            self.list_of_errors.append(nr_of_errors)
                
                
            if self.is_verbose:
                print('Epoch: {}, weights: {}, number of errors {}'.format(e, self.w, nr_of_errors))






X = np.array([
    [2,4,20],
    [4,3,-10],
    [5,6,13],
    [5,4,8],
    [3,4,5],
])


y = np.array([1,-1,-1,1,-1])

perc = Perceptron(epochs=150, is_verbose=True)
perc.fit(X, y)
print(perc.w)

#perc.predict(np.array([1,2,3,1]))
#perc.predict(np.array([2,2,8,1]))
#perc.predict(np.array([3,3,3,1]))
print(perc.w)


plt.scatter(range(perc.epochs), perc.list_of_errors)

