import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn import datasets
import random as rnd
sns.set()

cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS', 'RAD','TAX','PTRATIO','B','LSTAT','MEDV']


h = datasets.load_bost
housing = pd.read_csv(r'F:/\UdemyMachineLearning/\housing/\housing.data', sep=' +', engine='python', header=None, names=cols)
housing

# Wykres przedstawia srednią wartosc nieruchomosci w zaleznosci od jej lokalizacji przy rzecze (1)
plt.figure()
sns.barplot(x=housing['CHAS'],y=housing['MEDV'])
plt.show()

# Wykres przedstawia rozkład mediany wartości domów w zależności od bliskości do rzeki.
plt.figure()
sns.boxplot(x=housing['CHAS'],y=housing['MEDV'])
plt.show()
# Pudełko (Box): Reprezentuje przedział między pierwszym (Q1) a trzecim kwartylem (Q3) danych, czyli zawiera środkowe 50% wartości.
# Linia wewnątrz pudełka (Median Line): Reprezentuje medianę danych (Q2).
# Wąsy (Whiskers): Wskazują zakres danych, które nie są uważane za wartości odstające. Wąsy sięgają maksymalnie do 1.5 razy odległości między kwartylami (IQR) od Q1 i Q3.
# Kropki (Outliers): Reprezentują wartości odstające, które znajdują się poza zakresem wyznaczonym przez wąsy.



# Tworzy siatkę wykresów rozrzutu (scatter plots) i histogramów dla każdej pary zmiennych w zbiorze danych
plt.figure()
sns.pairplot(housing)
plt.show()



plt.figure()
sns.barplot(x=housing['CRIM'],y=housing['MEDV'])
plt.show()

corr_mat = np.corrcoef(housing.values.T)
corr_mat

# Przedstawia korelacje między zmiennymi
plt.figure()
sns.heatmap(data = corr_mat, 
            square=True, 
            cbar=True, 
            annot=True, 
            fmt='.2f', 
            annot_kws= {'size':5}, 
            xticklabels=housing.columns, 
            yticklabels=housing.columns
)
plt.show()


diamonds = pd.read_csv(r'F:/\UdemyMachineLearning/\diamonds/\diamonds.csv', usecols=['color','price'])
diamonds.head()

diam_copy = diamonds.copy()
diam_copy

diam_mean_org = diam_copy.groupby(by='color').mean()

missing_data = rnd.sample(range(0, len(diamonds)-1), 5)




diam_copy.loc[missing_data]
diam_copy.loc[missing_data, 'price'] = np.NaN

filter_nan = diam_copy['price'].isnull()

diam_copy.loc[filter_nan,'price2'] = diam_copy.loc[filter_nan,'color'].map(diam_mean_org['price'])

diam_copy.loc[filter_nan]
