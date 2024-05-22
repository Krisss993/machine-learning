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
import sys


class Perceptron:
    def __init__(self, eta, epochs, is_verbose=False):
        self.eta=eta
        self.epochs=epochs
        self.is_verbose=is_verbose
        
        
    def predict(self, X):
        ones = np.ones((X.shape[0],1))
        X_1 = np.append(X.copy(), ones, axis=1)
        y_pred = np.where(np.dot(X_1, self.w)>1,1,-1)
        return y_pred
        
    def __predict(self, X):
        y_pred = np.where(np.dot(X, self.w)>1,1,-1)
        return y_pred
    
    def fit(self, X, y):
        ones = np.ones((X.shape[0],1))
        X_1 = np.append(X.copy(), ones, axis=1)
        
        self.w = np.random.rand(X_1.shape[1])

        self.list_of_errors = []
        
        for e in range(self.epochs):
            y_pred = self.__predict(X_1)
            delta_w = self.eta*np.dot((y-y_pred), X_1)
            self.w+=delta_w
            
            error=np.where(y-y_pred!=0,1,0).sum()
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

perc = Perceptron(eta=0.1, epochs=100, is_verbose=True)

perc.fit(X, y)

perc.predict(X)


plt.scatter(range(perc.epochs), perc.list_of_errors)
plt.show()



lr=LR(eta=0.001, epochs=6, is_verbose=True)


lr.fit(X, y)

lr.predict(X)

plt.plot(range(lr.epochs), lr.list_of_errors)



wine = pd.read_csv(r'F:/\UdemyMachineLearning/winequality_/winequality-red.csv', sep=';')
wine                   
wine.info()
wine.describe()
wine.columns

plt.figure()
sns.heatmap(wine)
plt.show()

plt.figure()
sns.boxplot(x=wine['quality'], y=wine['alcohol'])
plt.show()

plt.figure()
plt.scatter(x=wine['quality'], y=wine['alcohol'])
plt.show()

plt.figure()
sns.pairplot(wine)
plt.show()
wine.corr()

X = wine[['volatile acidity','alcohol','sulphates','citric acid']].values
y = wine['quality'].values.reshape(-1,1)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2)

lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

lr.score(X_test, y_test)
lr.score(X_train, y_train)

len(X_test)
len(y_test)
plt.scatter(X_test[:,0], y_test)
plt.scatter(X_test[:,1], y_test)
plt.scatter(X_test[:,2], y_test)
plt.scatter(X_test[:,3], y_test)



class RANSAC:
    def __init__(self, min_acceptable_inliers=100, max_iters = 100, treshold=0.5):
        self.min_acceptable_inliers=min_acceptable_inliers
        self.treshold=treshold
        self.max_iters=max_iters
        
        self.best_model=None
        self.best_mask=None
        self.accepted_inliers_count = 0
        
    def fit(self, X, y, show_results = False):
        assert X.shape[1] == 1, '1'
        assert X.shape[0] >= 1.5 * self.min_acceptable_inliers
        
        data = np.hstack((X[:,0].reshape(-1,1), y.reshape(-1,1)))
        
        sample_size = data.shape[1]
        
        for i in range(self.max_iters):
            rand_idx = np.random.choice(len(data), size=sample_size, replace=False)
            points = data[rand_idx]
            
            a=(points[0,1]-points[1,1])/(points[0,0]-points[1,0]+sys.float_info.epsilon)
            b=points[0,1]-a*points[0,0]
            
            y_pred = a*data[:,0]+b

            this_inliers_mask = np.square(y-y_pred) < self.treshold
            this_inliers_count = np.sum(this_inliers_mask)
            better_counts = (this_inliers_count > self.accepted_inliers_count) and (this_inliers_count >= 1.5*self.min_acceptable_inliers)
            
            if better_counts:
                self.best_model=(a, b)
                self.best_mask=this_inliers_mask
                self.accepted_inliers_count = this_inliers_count
                
            if show_results:
                X_line = np.arange(X.min(), X.max())[:, np.newaxis]
                y_line = a*X_line+b
                plt.scatter(X[~this_inliers_mask], y[~this_inliers_mask],color='red')
                plt.scatter(X[this_inliers_mask], y[this_inliers_mask],color='green')
                plt.scatter(points[:,0], points[:,1], color='black')
                plt.plot(X_line, y_line, color='blue')
                #plt.legend(loc='lower right')
                plt.xlabel('Input')
                plt.ylabel('Response')
                plt.show()

cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS', 'RAD','TAX','PTRATIO','B','LSTAT','MEDV']


housing = pd.read_csv(r'F:/\UdemyMachineLearning/\housing/\housing.data', sep=' +', engine='python', header=None, names=cols)

X=housing['LSTAT'].values.reshape(-1,1)
X
y=housing['MEDV'].values
y
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
lr=LinearRegression()

lr.fit(X_train, y_train)

y_pred_test = lr.predict(X_test)
y_pred_train = lr.predict(X_train)

plt.figure()
plt.scatter(X_train, y_train)
plt.plot(X_train,y_pred_train, color='red')
plt.show()

plt.figure()
plt.scatter(X_test, y_test)
plt.plot(X_test,y_pred_test, color='red')
plt.show()
lr.score(X_test, y_test)

ran=RANSAC(treshold=6)

ran.fit(X_train, y_train, show_results=True)

