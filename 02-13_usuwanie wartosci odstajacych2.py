import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LinearRegression
from scipy import stats


wine = pd.read_csv(r'F:/\UdemyMachineLearning/winequality_/winequality-red.csv', sep=';')


plt.figure()
sns.boxplot(wine['alcohol'])
plt.plot()

# Z SCORE TO (WYNIK (X) - ŚREDNIA) / ODCHYLENIE STANDARDOWE
z= np.abs(stats.zscore(wine))
z

# 4 odchylenia standardowe
threshold=4
print(np.where(z>threshold))

wine_o_z = wine[(z<threshold).all(axis=1)]
wine_o_z

Q1 = wine.quantile(0.25)
Q1
Q3 = wine.quantile(0.75)
Q3
IQR = Q3-Q1
IQR

plt.scatter(abs(wine['alcohol'].values-wine['alcohol'].values.mean())/wine['alcohol'].values.std(),wine['quality'])

# CZY ISNIEJA WARTOSCI MNIEJSZE OD Q1 o 1.5 RAZA ROZSTĘP KWARTYLOWY LUB WIEKSZE OD Q3 o 1.5 IQR
((wine < (Q1 - 1.5 * IQR)) | (wine > (Q3 + 1.5 * IQR)))
# PRÓBKA
((wine < (Q1 - 1.5 * IQR)) | (wine > (Q3 + 1.5 * IQR))).iloc[14:18,4:7]

oultiner_condition = ((wine < (Q1 - 1.5 * IQR)) | (wine > (Q3 + 1.5 * IQR)))

wine_o_iqr = wine[~oultiner_condition.any(axis=1)]
wine_o_iqr

wine['quality'].unique()
wine_o_z['quality'].unique()
wine_o_iqr['quality'].unique()

for i in wine.columns[:-1]:
    fix, axs = plt.subplots(3,1, figsize=(10,7))
    sns.boxplot(x=wine['quality'], y=wine[i], ax=axs[0])
    sns.boxplot(x=wine_o_z['quality'], y=wine_o_z[i], ax=axs[1])
    sns.boxplot(x=wine_o_iqr['quality'], y=wine_o_iqr[i], ax=axs[2])
    plt.plot()


columns=wine.columns.drop('quality')

X=wine[columns]
y=wine['quality'].astype('float')

X=wine_o_z[columns]
y=wine_o_z['quality'].astype('float')

X=wine_o_iqr[columns]
y=wine_o_iqr['quality'].astype('float')

scaler = StandardScaler()
scaler.fit(X)
X_std = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2)

lr=LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

good_counter = np.count_nonzero(y_test==np.rint(y_pred))

total = len(y_test)

print(good_counter/total)

