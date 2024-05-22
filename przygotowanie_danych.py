
# Loading common data related modules
import numpy as np
import pandas as pd
import math 
 
# Loading modelling algorithms
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neighbors    import KNeighborsRegressor
from sklearn.ensemble     import AdaBoostRegressor
from sklearn.ensemble     import RandomForestRegressor
 
# Loading tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import r2_score
 
# Loading visualisation modules
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
 
# Configure visualisations 
%matplotlib inline

# Ignore warning messages
import warnings
warnings.filterwarnings('ignore')
 
#-----------------------------------------------------------------------------
# Read data
diamonds = pd.read_csv(r"F:\UdemyMachineLearning\diamonds\diamonds.csv")
 
#-----------------------------------------------------------------------------
# Review and clean the data (may be a repetitive task)
# remove unnecessary columns
diamonds.head()
diamonds.drop(['Unnamed: 0'] , axis=1 , inplace=True)
diamonds.head()
 
# review the data and get intuition about it
diamonds.shape
diamonds.info()
 
# find and eliminate nulls
diamonds.isnull().sum()

#TWORZY WYKRES z diamonds POKAZUJACY WARTOSCI null
msno.matrix(diamonds, figsize=(10,4)) # just to visualize. no missing values.

# search for illogical values
diamonds.describe()
diamonds.loc[(diamonds['x']==0) | (diamonds['y']==0) | (diamonds['z']==0)]
len(diamonds[(diamonds['x']==0) | (diamonds['y']==0) | (diamonds['z']==0)])

diamonds = diamonds[(diamonds[['x','y','z']] != 0).all(axis=1)]
# always check after execution
diamonds.loc[(diamonds['x']==0) | (diamonds['y']==0) | (diamonds['z']==0)]
 
# Detect dependencies in the data
# diamonds.corr daje macierz zaleznosci korelacji
corr = diamonds.corr(numeric_only=True)
corr
# sns.heatmap daje heatmap zaleznosci korelacji
sns.heatmap(data=corr, square=True , annot=True, cbar=True)
# wykresy korelacji dla kazdej zmiennej
sns.pairplot(diamonds)
#
# check distribution
#wykkres dystrybucji cla carat
sns.kdeplot(diamonds['carat'], fill=True , color='r')
#histogram dla carat
plt.hist(diamonds['carat'], bins=25)
#
# check correlation graph

sns.jointplot(x='carat' , y='price' , data=diamonds , size=5)
#
# analyze feature by feature, create hypotesis, try to find evidence
sns.catplot(x='cut', data=diamonds , kind='count',aspect=1)
sns.catplot(x='cut', y='price', data=diamonds, kind='box' ,aspect=1.5)
#
sns.catplot(x='color', data=diamonds , kind='count',aspect=1.5)
sns.catplot(x='color', y='price' , data=diamonds , kind='violin', aspect=1.5)
#
# try to use different visualisation methods
sns.catplot(x='clarity', data=diamonds , kind='count',aspect=1.5)
sns.catplot(x='clarity', y='price' , data=diamonds , kind='violin', aspect=1.5)
#
labels = diamonds.clarity.unique().tolist()
labels
sizes = diamonds.clarity.value_counts().tolist()
sizes
colors = ['#006400', '#E40E00', '#A00994', '#613205', '#FFED0D', '#16F5A7','#ff9999','#66b3ff']
explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=0)
plt.axis('equal')
plt.title("Percentage of Clarity Categories")
plt.plot()
fig=plt.gcf()
fig.set_size_inches(6,6)
plt.show()
#
# try to find specific groups/classifications - repetitive process
sns.boxplot(x='clarity', y='price', data=diamonds)
#
plt.hist('depth' , data=diamonds , bins=50)
sns.jointplot(x='depth', y='price', data=diamonds, size=5)
#
sns.kdeplot(diamonds['table'], fill=True , color='orange')
sns.jointplot(x='table', y='price', data=diamonds , size=5)
 
#-----------------------------------------------------------------------------
# feature engineering - ananlyzing separately xyz doesn't make sense
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
#
sns.jointplot(x='volume', y='price' , data=diamonds, size=5)
#
diamonds.drop(['x','y','z'], axis=1, inplace= True)
diamonds.head()
diamonds.corr(numeric_only=True)
#
# One hot encoding
diamonds = pd.get_dummies(diamonds, prefix_sep='_', drop_first=True)
diamonds.head()
 
#-----------------------------------------------------------------------------
# splitting data into features X, and labels y
X = diamonds.drop(['price'], axis=1)
y = diamonds['price']
#
# splitting data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=66)
 
#-----------------------------------------------------------------------------
# scaling values
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train
 
 
#-----------------------------------------------------------------------------
# test different algorithms to get the data predictions 
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




























w = np.array([0.38544573,0.40046814,0.50987714,0.94058191])
w

X1 = np.array([2,4,20,1])
Z1 = np.dot(X1, w)
Z1

X2 = np.array([4,3,-10,1])
Z2 = np.dot(X2, w)
Z2




delta_w = 0.01 * (1 - 1) * X1
delta_w


 