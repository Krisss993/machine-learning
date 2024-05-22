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

cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS', 'RAD','TAX','PTRATIO','B','LSTAT','MEDV']
housing = pd.read_csv(r'F:/\UdemyMachineLearning/\housing/\housing.data', sep=' +', engine='python', header=None, names=cols)

data = housing[['LSTAT', 'MEDV']]
data.info()

z = stats.zscore(data)
z
treshold=3
data_z = data[(z<treshold).all(axis=1)]
data_z

Q1=data.quantile(0.25)
Q3=data.quantile(0.75)
IQR=Q3-Q1
outliers = (data<(Q1-1.5*IQR)) | (data>(Q3+1.5*IQR))
outliers

data_iqr = data[~outliers.any(axis=1)]
data_iqr

X=data['LSTAT'].values.reshape(-1,1)
y=data['MEDV'].values.reshape(-1,1)

Xz=data_z['LSTAT'].values.reshape(-1,1)
yz=data_z['MEDV'].values.reshape(-1,1)

Xiqr=data_iqr['LSTAT'].values.reshape(-1,1)
yiqr=data_iqr['MEDV'].values.reshape(-1,1)

plt.scatter(X,y)
plt.scatter(Xz,yz)
plt.scatter(Xiqr,yiqr)

scaler = StandardScaler()
scaler.fit(X,y)
X=scaler.transform(X)

scaler.fit(Xz,yz)
Xz=scaler.transform(Xz)

scaler.fit(Xiqr,yiqr)
Xiqr=scaler.transform(Xiqr)





X_train, X_test, y_train, y_test = train_test_split(Xiqr, yiqr, test_size=0.2)
lr=LinearRegression()

lr.fit(X_train, y_train)
y_pred_train = lr.predict(X_train)

plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, y_pred_train, color='red')


mae_train = mean_absolute_error(y_train, lr.predict(X_train))
mae_test = mean_absolute_error(y_test, lr.predict(X_test))
print("MAE TRAIN {}, TEST {}".format(round(mae_train,2), round(mae_test,2)))
 
mse_train = mean_squared_error(y_train, lr.predict(X_train))
mse_test = mean_squared_error(y_test, lr.predict(X_test))
print("MSE TRAIN {}, TEST {}".format(round(mse_train,2), round(mse_test,2)))
 
r2_train = r2_score(y_train, lr.predict(X_train))
r2_test = r2_score(y_test, lr.predict(X_test))
print("R2  TRAIN {}, TEST {}".format(round(r2_train,2), round(r2_test,2)))


