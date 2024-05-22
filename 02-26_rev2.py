import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LinearRegression
from scipy import stats
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


class Perceptron:
    def __init__(self, eta=0.01, epochs=150, is_verbose=False):
        self.eta=eta
        self.epochs=epochs
        self.is_verbose=is_verbose
        self.list_of_errors = []
        
    def predict(self, X):
        ones=np.ones((X.shape[0],1))
        X_1=np.append(X.copy(), ones, axis=1)
        y_pred = np.where(self.get_activation(X_1)>0,1,-1)
        return y_pred
    
    def get_activation(self, X):
        activation = np.dot(X, self.w)
        return activation
    
    def fit(self,X,y):
        ones = np.ones((X.shape[0],1))
        X_1 = np.append(X, ones, axis=1)
        self.w = np.random.rand(X_1.shape[1])
        self.list_of_errors = []
        self.list_of_errors = []
        for e in range(self.epochs):
            error=0
            activation=self.get_activation(X_1)
            delta_w = self.eta * np.dot((y-activation),X_1)
            self.w+=delta_w
            error = np.square(y - activation).sum()/2.0
            
            self.list_of_errors.append(error)
            
            
            if (self.is_verbose):
                print('Epochs: {}, weights: {}, errors {}'.format(e, self.w, error))
        

class LR:
    def __init__(self, eta=0.01, epochs=150, is_verbose=False):
        self.eta=eta
        self.epochs=epochs
        self.is_verbose=is_verbose
        self.list_of_errors = []
        
    def predict(self, X):
        ones=np.ones((X.shape[0],1))
        X_1=np.append(X.copy(), ones, axis=1)
        return self.get_activation(X_1)
    
    def get_activation(self, X):
        activation = np.dot(X, self.w)
        return activation
    
    def fit(self,X,y):
        ones = np.ones((X.shape[0],1))
        X_1 = np.append(X, ones, axis=1)
        self.w = np.random.rand(X_1.shape[1])

        self.list_of_errors = []
        for e in range(self.epochs):
            error=0
            activation=self.get_activation(X_1)
            delta_w = self.eta * np.dot((y-activation),X_1)
            self.w+=delta_w
            error = np.square(y - activation).sum()/2.0
            
            self.list_of_errors.append(error)
            
            
            if (self.is_verbose):
                print('Epochs: {}, weights: {}, errors {}'.format(e, self.w, error))




X = np.array([
    [2,4,20],
    [4,3,-10],
    [5,6,13],
    [5,4,8],
    [3,4,5],
])

y = np.array([1,-1,-1,1,-1])

perc = Perceptron(eta=0.001, epochs=100, is_verbose=True)

perc.fit(X, y)

perc.predict(X)



wine = pd.read_csv(r'F:/\UdemyMachineLearning/winequality_/winequality-red.csv', sep=';')
wine                   
wine.info()
wine.describe()
wine.columns



plt.figure()
sns.boxplot(x=wine['quality'], y=wine['alcohol'])
plt.plot()
plt.scatter(x=wine['quality'], y=wine['alcohol'])

X=wine['alcohol'].values.reshape(-1,1)
y=wine['quality'].values


scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

lr=LinearRegression()

lr.fit(X_train, y_train)
y_pred_test = lr.predict(X_test)



data=wine[['alcohol','quality']]

z = stats.zscore(data)
z
treshold=3

data_z = data[(z<treshold).all(axis=1)]
data_z

Q1=data.quantile(0.25)
Q1
Q3=data.quantile(0.75)
Q3

IQR=Q3-Q1

outliers = (data<Q1-1.5*IQR) | (data>Q3+1.5*IQR)
data_iqr = data[~outliers.any(axis=1)]
data_iqr


X=data_iqr['alcohol'].values.reshape(-1,1)
y=data_iqr['quality'].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

lr=LinearRegression()

lr.fit(X_train, y_train)
y_pred_test = lr.predict(X_test)

plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred_test, color='red')


mae = mean_absolute_error(y_test, y_pred_test)
mae
mse = mean_squared_error(y_test, y_pred_test)
mse

lr.score(X_test, y_test)
