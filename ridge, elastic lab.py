import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LinearRegression, Ridge, Lasso, ElasticNet
from scipy import stats
import sys
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


wine = pd.read_csv(r'F:/\UdemyMachineLearning/winequality_/winequality-red.csv', sep=';')
wine.columns

X=wine.drop('quality', axis=1).values
X

y=wine['quality'].values.astype('float')
y

scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LinearRegression()
lr.fit(X_train, y_train)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

ridge=Ridge(alpha=0.1)
ridge.fit(X_train, y_train)

elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train, y_train)

coef = pd.DataFrame([lr.coef_, lasso.coef_, ridge.coef_, elastic.coef_]).transpose()
coef
coef.index=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']

coef.columns=['LR', 'Lasso', 'Ridge', 'Elastic']

coef
columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']


fig, axs = plt.subplots(4, figsize=(10,10))
fig.suptitle('Coef. in diff. reg.')

linear_coef = pd.Series(lr.coef_, columns)
axs[0].bar(x=lr.coef_, height=linear_coef)

lasso_coef = pd.Series(lasso.coef_, columns).sort_values()
axs[1].bar(x=lasso.coef_, height=lasso_coef)

ridge_coef = pd.Series(ridge.coef_, columns).sort_values()
axs[2].bar(x=ridge.coef_, height=ridge_coef)

elastic_coef = pd.Series(elastic.coef_, columns).sort_values()
axs[3].bar(x=elastic.coef_, height=elastic_coef)

fig.show()

r2=pd.DataFrame(columns=['Train', 'Test'], index= ['lr', 'lasso', 'ridge', 'elastic net'])

r2.loc['lr'] = [r2_score(y_train, lr.predict(X_train)),
                    r2_score(y_test, lr.predict(X_test))]
r2.loc['lasso'] = [r2_score(y_train, lasso.predict(X_train)),
                    r2_score(y_test, lasso.predict(X_test))]
r2.loc['ridge'] = [r2_score(y_train, ridge.predict(X_train)),
                    r2_score(y_test, ridge.predict(X_test))]
r2.loc['elastic net'] = [r2_score(y_train, elastic.predict(X_train)),
                    r2_score(y_test, elastic.predict(X_test))]
r2




alpha_list = np.arange(0.01, 1, 0.01 )
alpha_list

lasso_coef = pd.DataFrame(columns = columns)
lasso_r2 = []

for a in alpha_list:
    lasso = Lasso(alpha=a)
    lasso.fit(X_train, y_train)
    lasso_coef.loc[a] = lasso.coef_
    lasso_r2.append(r2_score(y_train, lasso.predict(X_train)))


f = plt.figure(figsize=(10,5))
plt.title('Lasso coef. changes')
lasso_coef.plot(kind='line',ax=f.gca())
plt.legend(loc='upper right')
plt.show()

plt.plot(alpha_list, lasso_r2)
plt.title('Lasso R2 score')
plt.show()


ridge_coef = pd.DataFrame(columns=columns)
ridge_r2 = []

for a in alpha_list:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train, y_train)
    ridge_coef.loc[a] = ridge.coef_
    ridge_r2.append(r2_score(y_train, ridge.predict(X_train)))
    
f = plt.figure(figsize=(10,5))
plt.title('Ridge coef. changes')
ridge_coef.plot(kind='line', ax=f.gca())
plt.show()

plt.plot(alpha_list, ridge_r2)
plt.title('Ridge R2 score')
plt.show()



elastic_coef = pd.DataFrame(columns=columns)
elastic_r2 = []

for a in alpha_list:
    elastic = ElasticNet(alpha=a)
    elastic.fit(X_train, y_train)
    elastic_coef.loc[a] = elastic.coef_
    elastic_r2.append(r2_score(y_train, elastic.predict(X_train)))
    
f = plt.figure(figsize=(10,5))
plt.title('Elastic coef. change')
elastic_coef.plot(kind='line', ax=f.gca())
plt.show()

plt.plot(alpha_list, elastic_r2)
plt.title('Elastic R2 score')
plt.show()




















cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS', 'RAD','TAX','PTRATIO','B','LSTAT','MEDV']


housing = pd.read_csv(r'F:/\UdemyMachineLearning/\housing/\housing.data', sep=' +', engine='python', header=None, names=cols)

X=housing.drop('MEDV', axis=1)
X
y=housing['MEDV'].values
y


scaler=StandardScaler()
scaler.fit(X)
X=scaler.transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr=LinearRegression()
lr.fit(X_train, y_train)
lr.coef_
lr.score(X_test, y_test)
lr.score(X_train, y_train)

lasso_df = pd.DataFrame({
    'param_value': np.arange(0.01, 1, 0.1),
    'r2_result': 0.,
    'number_of_features':0
    })


for i in range(lasso_df.shape[0]):
    al = lasso_df.loc[i, 'param_value']
    lasso = Lasso(alpha=al)
    lasso.fit(X_train, y_train)
    r2 = r2_score(y_test, lasso.predict(X_test))
    lasso_df.loc[i, 'r2_result'] = r2
    lasso_df.loc[i, 'number_of_features'] = len(lasso.coef_[ lasso.coef_ > 0])




ridge_df = pd.DataFrame({
    'param_value':np.arange(0.01, 1, 0.1),
    'r2_result': 0.,
    'number_of_features':0
    })


for i in range(ridge_df.shape[0]):
    al = ridge_df.loc[i, 'param_value']
    ridge = Ridge(alpha=al)
    ridge.fit(X_train, y_train)
    r2 = r2_score(y_test, ridge.predict(X_test))
    ridge_df.loc[i, 'r2_result'] = r2
    ridge_df.loc[i, 'number_of_features'] = len(ridge.coef_[ridge.coef_>0])

















elastic_df = pd.DataFrame({
    'param_value':np.arange(0.01, 1, 0.1),
    'r2_result': 0.,
    'number_of_features':0
    })

for i in range(elastic_df.shape[0]):
    al = elastic_df.loc[i,'param_value']
    model = ElasticNet(alpha=al)
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    elastic_df.loc[i,'r2_result'] = r2
    elastic_df.loc[i,'number_of_features'] = len(model.coef_[model.coef_>0])





fig, axs = plt.subplots(3, figsize=(10,10))
axs[0].title.set_text('Lasso')
axs[0].scatter(x = lasso_df['param_value'], y=lasso_df['r2_result']*10)
axs[0].scatter(x = lasso_df['param_value'], y=lasso_df['number_of_features'])
axs[1].title.set_text('Ridge')
axs[1].scatter(x = ridge_df['param_value'], y = ridge_df['r2_result']*10)
axs[1].scatter(x = ridge_df['param_value'], y = ridge_df['number_of_features'])
axs[2].title.set_text('Elastic')
axs[2].scatter(x = elastic_df['param_value'], y = elastic_df['r2_result']*10)
axs[2].scatter(x = elastic_df['param_value'], y = elastic_df['number_of_features'])

fig.show()











































































