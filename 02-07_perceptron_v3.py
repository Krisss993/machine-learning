import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Perceptron:
    def __init__(self, eta=0.1, epochs=50, is_verbose=False):
        self.eta=eta
        self.epochs=epochs
        self.is_verbose=is_verbose
        
        
    def predict(self,x):
        ones = np.ones((x.shape[0],1))
        x_1 = np.append(x.copy(), ones, axis=1)
        activation = self.get_activation(x_1)
        y_pred = np.where(activation > 0, 1, -1)
        return y_pred
    
    def get_activation(self,x):
        activation = np.dot(x, self.w)
        return activation
    
    
    
    
    def fit(self,X,y):
        ones = np.ones((X.shape[0],1))
        X_1 = np.append(X.copy(), ones, axis=1)
        self.w = np.random.rand(X_1.shape[1])
        self.list_of_errors = []
        error = 0
        
        for e in range(self.epochs):
            activation = self.get_activation(X_1)
            
            delta_w = self.eta * np.dot((y-activation),X_1)
            self.w+=delta_w
            error = np.square(y-activation).sum()/2.0
            self.list_of_errors.append(error)
            
        
            if (self.is_verbose):
                print('Epoch: {}, weights: {}, error {}'.format(e, self.w, error))
    
    







X = np.array([
    [2,4,20],
    [4,3,-10],
    [5,6,13],
    [5,4,8],
    [3,4,5],
])


y = np.array([1,-1,-1,1,-1])

perc = Perceptron(eta=0.0001,epochs=100, is_verbose=True)
perc.fit(X, y)

print(perc.get_activation([2,4,20,1]))

# plt.scatter(range(perc.epochs),perc.list_of_errors)

df = pd.read_csv(r'F:\UdemyMachineLearning\iris\iris.data', header = None)
df = df[:100]
X = df.iloc[ :100, :-1].values
X
df[4] = df[4].apply(lambda x:1 if x == 'Iris-setosa' else -1)
y=df[4].values
y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

p = Perceptron(eta = 0.0001, epochs=100) 
p.fit(X_train, y_train)   
             
y_pred = p.predict(X_test)
 
plt.scatter(range(p.epochs), p.list_of_errors)
