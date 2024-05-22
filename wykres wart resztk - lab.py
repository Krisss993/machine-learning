import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LinearRegression
from scipy import stats


cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS', 'RAD','TAX','PTRATIO','B','LSTAT','MEDV']
housing = pd.read_csv(r'F:/\UdemyMachineLearning/\housing/\housing.data', sep=' +', engine='python', header=None, names=cols)

data=housing.loc[:,['LSTAT','MEDV']]

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3-Q1

outliners = (data < (Q1-IQR*1.5)) | (data > (Q3+IQR*1.5))
outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 +1.5 * IQR)))

data_iqr = data[~outliners.any(axis=1)]
data_iqr2 = data[~outliers.any(axis=1)]

lr = LinearRegression()
X = data['LSTAT'].values.reshape(-1,1)
y = data['MEDV'].values.reshape(-1,1)

scaler = StandardScaler()
scaler.fit(X)
std_X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(std_X, y, test_size=0.2)

lr.fit(X_train, y_train)

y_test_pred = lr.predict(X_test)

y_train_pred = lr.predict(X_train)

plt.scatter(X_train, y_train)
plt.plot(X_train, y_train_pred, color='red')


plt.scatter(X_test, y_test)
plt.plot(X_test, y_test_pred, color='red')

fig, ax = plt.subplots(1,2,figsize=(8,4))
ax[0].scatter(y_train, y_train_pred - y_train, s=80, facecolors='none', edgecolors='b')
ax[1].scatter(y_test, y_test_pred - y_test, s=80, facecolors='none', edgecolors='r')
fig.show()

X = data_iqr['LSTAT'].values.reshape(-1,1)
y = data_iqr['MEDV'].values.reshape(-1,1)
