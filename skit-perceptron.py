import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
%matplotlib inline


iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.3)

sc = StandardScaler()
sc.fit(X)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

perc = Perceptron(max_iter=20, eta0=0.1)

perc.fit(X_train_std, y_train)

y_pred = perc.predict(X_test_std)

y_test
y_pred

# ERRORS
bad_results = [(i,j) for (i,j) in zip(y_pred[y_pred!=y_test], y_test[y_pred!=y_test])]
good_results = [(i,j) for (i,j) in zip(y_pred[y_pred==y_test], y_test[y_pred==y_test])]

good_results

# OCENA MODELU
print(perc.score(X_test_std, y_test))

# WAGI
print(perc.coef_)

# GRANICA DEZYCYJNA 
print(perc.intercept_)

# LICZBA WYKONANYCH EPOK
print(perc.n_iter_)

# LICZBA AKTUALIZACJI WAG
print(perc.t_)
