import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LinearRegression


wine = pd.read_csv(r'F:/\UdemyMachineLearning/winequality_/winequality-red.csv', sep=';')
wine                   
wine.info()
wine.describe()
wine.columns



plt.figure()
sns.boxplot(x=wine['quality'], y=wine['alcohol'])
plt.plot()
plt.scatter(x=wine['quality'], y=wine['alcohol'])

for i in wine.columns:
    print(wine[i].describe())
    
for i in wine.columns[:-1]:
    plt.figure()
    sns.boxplot(x=wine['quality'], y=wine[i])
    plt.plot()
    
    
for i in wine.columns[:-1]:
    plt.figure()
    sns.barplot(x=wine['quality'], y=wine[i])
    plt.plot()
    
# MACZIERZ KORELACJI
corr_matrix = np.corrcoef(wine.values.T)
corr_matrix

fig, ax = plt.subplots(figsize=(11,11))
sns.set(font_scale=1.1)
sns.heatmap(data = corr_matrix, 
            square=True, 
            cbar=True, 
            annot=True, 
            fmt='.2f', 
            annot_kws= {'size':10}, 
            xticklabels=wine.columns, 
            yticklabels=wine.columns
)

sns.heatmap(data = corr_matrix)

sns.pairplot(wine, height=1.5)

columns = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid', 'total sulfur dioxide', 'density','quality']

sns.pairplot(wine[columns], size=1.5)


