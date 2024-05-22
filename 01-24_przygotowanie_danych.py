import numpy as np
import pandas as pd
import math 
 

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neighbors    import KNeighborsRegressor
from sklearn.ensemble     import AdaBoostRegressor
from sklearn.ensemble     import RandomForestRegressor
 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import r2_score
 

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
 

import warnings
warnings.filterwarnings('ignore')
 
#-----------------------------------------------------------------------------

diamonds = pd.read_csv(r"F:\UdemyMachineLearning\diamonds\diamonds.csv")
 
#-----------------------------------------------------------------------------
# usuniecie niepotrzebnych kolumn
diamonds.head()
diamonds.drop(['Unnamed: 0'] , axis=1 , inplace=True)
diamonds.head()
 
# kształt i informacje o danych
diamonds.shape
diamonds.info()
diamonds.columns

# usuniecie wartosci nan/null
diamonds.isnull().sum()

#TWORZY WYKRES z diamonds POKAZUJACY WARTOSCI null
msno.matrix(diamonds, figsize=(10,4))

# szukanie wartosci nielogicznych
diamonds.describe()
diamonds.loc[(diamonds['x']==0) | (diamonds['y']==0) | (diamonds['z']==0)]
len(diamonds[(diamonds['x']==0) | (diamonds['y']==0) | (diamonds['z']==0)])

# usunięcie tych wartosci
diamonds = diamonds[(diamonds[['x','y','z']] != 0).all(axis=1)]
# sprawdzenie
diamonds.loc[(diamonds['x']==0) | (diamonds['y']==0) | (diamonds['z']==0)]
 
# Wykrywanie zależności w danych
# diamonds.corr tworzy macierz zaleznosci korelacji
corr = diamonds.corr(numeric_only=True)
corr
# sns.heatmap tworzy heatmape zaleznosci korelacji
plt.figure()
sns.heatmap(data=corr, square=True , annot=True, cbar=True)
plt.show()

# wykresy korelacji dla kazdej zmiennej
plt.figure()
sns.pairplot(diamonds)
plt.show()

#
# sprawdzenie rozkladu
# wykres dystrybucji cla carat
plt.figure()
sns.kdeplot(diamonds['carat'], fill=True , color='r')
plt.show()

# histogram dla carat
plt.figure()
plt.hist(diamonds['carat'], bins=25)
plt.show()
#
# Wykres korelacji pomiedzy carat i price
plt.figure()
sns.jointplot(x='carat' , y='price' , data=diamonds , size=5)
plt.show()
#

# Wykresy dla poszczegolnych zmiennych
plt.figure()
sns.catplot(x='cut', data=diamonds , kind='count',aspect=1)
plt.show()
#
plt.figure()
sns.catplot(x='cut', y='price', data=diamonds, kind='box' ,aspect=1.5)
plt.show()
#
plt.figure()
sns.catplot(x='color', data=diamonds , kind='count',aspect=1.5)
plt.show()
#
plt.figure()
sns.catplot(x='color', y='price' , data=diamonds , kind='violin', aspect=1.5)
plt.show()
#
plt.figure()
sns.catplot(x='clarity', data=diamonds , kind='count',aspect=1.5)
plt.show()
#
plt.figure()
sns.catplot(x='clarity', y='price' , data=diamonds , kind='violin', aspect=1.5)
plt.show()
#

# labelki do wykresu kolowego zmiennej clarity
labels = diamonds.clarity.unique().tolist()
labels

# rozmiar danych
sizes = diamonds.clarity.value_counts().tolist()
sizes

colors = ['#006400', '#E40E00', '#A00994', '#613205', '#FFED0D', '#16F5A7','#ff9999','#66b3ff']

# parametr explode "wysuwa" fragment wykresu z zamknietego kola o okreslona wartosc
explode = (0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1)

plt.figure()
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=0)
plt.axis('equal')
plt.title("Percentage of Clarity Categories")
plt.plot()
fig=plt.gcf()
fig.set_size_inches(6,6)
plt.show()
#
# 
plt.figure()
sns.boxplot(x='clarity', y='price', data=diamonds)
plt.show()
#
plt.figure()
plt.hist('depth' , data=diamonds , bins=50)
sns.jointplot(x='depth', y='price', data=diamonds, size=5)
plt.show()
#
plt.figure()
sns.kdeplot(diamonds['table'], fill=True , color='orange')
sns.jointplot(x='table', y='price', data=diamonds , size=5)
plt.show()
 
#-----------------------------------------------------------------------------
# inżynieria cech
plt.figure()
sns.kdeplot(diamonds['x'] , fill=True , color='r' )
sns.kdeplot(diamonds['y'] , fill=True , color='g' )
sns.kdeplot(diamonds['z'] , fill= True , color='b')
plt.xlim(2,10)

diamonds['volume'] = diamonds['x']*diamonds['y']*diamonds['z']
diamonds.head()
#
plt.figure(figsize=(5,5))
plt.hist( x=diamonds['volume'] , bins=30 ,color='g')
plt.xlabel('Volume in mm^3')
plt.ylabel('Frequency')
plt.title('Distribution of Diamond\'s Volume')
plt.xlim(0,1000)
plt.ylim(0,50000)
plt.show()
#
plt.figure()
sns.jointplot(x='volume', y='price' , data=diamonds, size=5)
plt.show()
#
diamonds.drop(['x','y','z'], axis=1, inplace= True)
diamonds.head()
diamonds.corr(numeric_only=True)
#
# podzielenie
diamonds = pd.get_dummies(diamonds, prefix_sep='_', drop_first=True)
diamonds.head()
 
#-----------------------------------------------------------------------------
# podzielenie danych na zmienne objasniajace i zmienne wynikowe
X = diamonds.drop(['price'], axis=1)
y = diamonds['price']
#
# podziala danych na zbior testowy oraz treningowy
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=66)
 
#-----------------------------------------------------------------------------
# skalowanie
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train
 
 
#-----------------------------------------------------------------------------
# Testowanie różnych algorytmy, aby uzyskać najlepsze prognozy danych
scores = []
models = ['Linear Regression', 'Lasso Regression', 'AdaBoost Regression', 
          'Ridge Regression', 'RandomForest Regression', 
          'KNeighbours Regression']
 
#-----------------------------------------------------------------------------
# Linear regression
lr = LinearRegression()
lr.fit(X_train , y_train)
y_pred = lr.predict(X_test)
r2 = r2_score(y_test, y_pred)
 
scores.append(r2)
print('Linear Regression R2: {0:.2f}'.format(r2))
 
 
# Lasso
lasso = Lasso()
lasso.fit(X_train , y_train)
y_pred = lasso.predict(X_test)
r2 = r2_score(y_test, y_pred)
 
scores.append(r2)
print('Lasso Regression R2: {0:.2f}'.format(r2))
 
 
# Adaboost classifier
adaboost = AdaBoostRegressor(n_estimators=1000)
adaboost.fit(X_train , y_train)
y_pred = adaboost.predict(X_test)
r2 = r2_score(y_test, y_pred)
 
scores.append(r2)
print('AdaBoost Regression R2: {0:.2f}'.format(r2))
 
# Ridge
ridge = Ridge()
ridge.fit(X_train , y_train)
y_pred = ridge.predict(X_test)
r2 = r2_score(y_test, y_pred)
 
scores.append(r2)
print('Ridge Regression R2: {0:.2f}'.format(r2))
 
 
# Random forest
randomforest = RandomForestRegressor()
randomforest .fit(X_train , y_train)
y_pred = randomforest .predict(X_test)
r2 = r2_score(y_test, y_pred)
 
scores.append(r2)
print('Random Forest R2: {0:.2f}'.format(r2))
 
 
# K-Neighbours
kneighbours = KNeighborsRegressor()
kneighbours.fit(X_train , y_train)
y_pred = kneighbours.predict(X_test)
r2 = r2_score(y_test, y_pred)
 
scores.append(r2)
print('K-Neighbours Regression R2: {0:.2f}'.format(r2))
 
 
#-----------------------------------------------------------------------------
ranking = pd.DataFrame({'Algorithms' : models , 'R2-Scores' : scores})
ranking = ranking.sort_values(by='R2-Scores' ,ascending=False)
ranking
 
sns.barplot(x='R2-Scores' , y='Algorithms' , data=ranking)











 