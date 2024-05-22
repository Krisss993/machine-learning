import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LinearRegression
from scipy import stats
import sys
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score



class RANSAC1:
    def __init__(self, max_iters_k=100, treshold=0.5, min_acceptable_inliners=100):
        self.max_iters_k=max_iters_k
        self.treshold=treshold
        self.min_acceptable_inliners=min_acceptable_inliners
        
        self.best_model=None
        self.best_inliner_count=0
        self.inliner_mask=None
        
        
    def fit(self, X, y, show_partial_results=False):
        # SPRAWDZENIE CZY TABLICA JEST JEDNOWYMIAROWA
        assert X.shape[1] == 1, 'The input matrix should have only one dimension'
        assert X.shape[0] > 1.5 * self.min_acceptable_inliners, 'The input matrix should be 1.5 greater than inliners'
        
        self.best_model=None
        self.best_inliner_count=0
        self.inliner_mask=None
        
        sample_size = X.shape[1] + 1
        
        data = np.hstack((X[:,0].reshape(-1,1), y.reshape(-1,1)))
        
        for i in range(self.max_iters_k):
            #WYZNACZAMY LOSOWE PUNKTY PRZEZ KTORE BEDZIE PRZECHODZIC PONIZSZA LINIA
            #RANDOMLY CHOOSE POINTS
            rand_idx = np.random.choice(len(data), size = sample_size, replace=False)
            points = data[rand_idx,:]
            
            #DETERMINE THE LINE EQUATION
            #WYZNACZAMY ROWNANIE LINII PRZECHODZACEJ PRZEZ 2 WYLOSOWANE WYZEJ PUNKTY
            a=(points[0,1]-points[1,1])/(points[0,0]-points[1,0]+sys.float_info.epsilon)
            b=points[0,1]-a*points[0,0]
            
            #CALCULATE PREDICTED POINTS
            y_pred = a*data[:,0]+b
            
            #DETERMINE POINTS WITHIN TRESHOLD
            #CZY WSZYSTKIE PUNKTY ZE ABIORU DANYCH SA ODDALONE OD PROSTEJ O ODLEGLOSC MNIEJSZA NIZ TRESHOLD
            this_inliner_mask = np.square(y_pred-y)<self.treshold
            this_inliner_count=np.sum(this_inliner_mask)
            
            # SPRAWDZAMY CZY MODEL JEST LEPSZY OD WCZESNIEJ ZNALEZIONYCH, LEPSZY JEST TEN MODEL KTÓRY MA WIECEJ INLIEROW
            better_found = ((this_inliner_count > self.min_acceptable_inliners) and (this_inliner_count > self.best_inliner_count))
            
            if better_found:
                self.best_model=(a,b)
                self.best_inliner_count=this_inliner_count
                self.inliner_mask=this_inliner_mask
                
            if show_partial_results:
                line_X = np.arange(X.min(), X.max())[:, np.newaxis]
                line_y = a*line_X + b
                
                #PUNKTY KTORE W INLINER MASK MAJA WARtoSC truE
                plt.scatter(X[this_inliner_mask], y[this_inliner_mask], color='green', marker='.', label='Inliers')
                
                #PUNKTY KTORE W INLINER MASK MAJA WARtoSC FALSE
                plt.scatter(X[~this_inliner_mask], y[~this_inliner_mask], color='red', marker='.', label='Outliers')
                
                #PROSTA ODPOWIADAJACA AX+B
                plt.plot(line_X, line_y, color='blue',linewidth=2, label='RANSC')
                
                #KROPKI OZNACZAJACE WYLOSOWANE PUNKTY
                plt.scatter(points[:,0], points[:,1], color='black', marker='o', label='Sampled points',s=100)
                
                
                plt.legend(loc='lower right')
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

plt.scatter(X_train, y_train)
plt.plot(X_train,y_pred_train, color='red')

plt.scatter(X_test, y_test)
plt.plot(X_test,y_pred_test, color='red')
lr.score(X_test, y_test)

ran=RANSAC1()

ran.fit(X_train, y_train, show_partial_results=True)

data = housing[['LSTAT','MEDV']]
z = stats.zscore(data)
z
treshold=3
data_z = data[(z<treshold).all(axis=1)]
data_z


X=data_z['LSTAT'].values.reshape(-1,1)
X
y=data_z['MEDV'].values

scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
lr=LinearRegression()

lr.fit(X_train, y_train)

y_pred_test = lr.predict(X_test)
y_pred_train = lr.predict(X_train)

plt.scatter(X_train, y_train)
plt.plot(X_train,y_pred_train, color='red')

plt.scatter(X_test, y_test)
plt.plot(X_test,y_pred_test, color='red')
lr.score(X_test, y_test)


Q1=data.quantile(0.25)
Q3=data.quantile(0.75)
IQR=Q3-Q1




oultiner_condition = ((data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR))

data_iqr = data[~oultiner_condition.any(axis=1)]
data_iqr


X=data_iqr['LSTAT'].values.reshape(-1,1)
X
y=data_iqr['MEDV'].values

scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
lr=LinearRegression()

lr.fit(X_train, y_train)

y_pred_test = lr.predict(X_test)
y_pred_train = lr.predict(X_train)

plt.scatter(X_train, y_train)
plt.plot(X_train,y_pred_train, color='red')

plt.scatter(X_test, y_test)
plt.plot(X_test,y_pred_test, color='red')
lr.score(X_test, y_test)
lr.score(X_train, y_train)

mae_train = mean_absolute_error(y_train, lr.predict(X_train))
mae_test = mean_absolute_error(y_test, lr.predict(X_test))
print("MAE TRAIN {}, TEST {}".format(round(mae_train,2), round(mae_test,2)))
 
mse_train = mean_squared_error(y_train, lr.predict(X_train))
mse_test = mean_squared_error(y_test, lr.predict(X_test))
print("MSE TRAIN {}, TEST {}".format(round(mse_train,2), round(mse_test,2)))
 
r2_train = r2_score(y_train, lr.predict(X_train))
r2_test = r2_score(y_test, lr.predict(X_test))
print("R2  TRAIN {}, TEST {}".format(round(r2_train,2), round(r2_test,2)))




















ransac = RANSAC1(max_iters_k = 100, treshold = 20, min_acceptable_inliners = 100)
 
ransac.fit(X, y, show_partial_results=True)


X_train, X_test, y_train, y_test = train_test_split(X[ransac.inliner_mask], y[ransac.inliner_mask], test_size=0.2)
lr=LinearRegression()
lr.fit(X_train, y_train)
y_pred_test = lr.predict(X_test)
y_pred_train = lr.predict(X_train)


line_X = np.arange(X.min(), X.max())[:, np.newaxis]
line_y = ransac.best_model[0] * line_X + ransac.best_model[1]
 
plt.figure(figsize=(7,5))
plt.scatter(X[ransac.inliner_mask], y[ransac.inliner_mask], color='green', marker='.', label='Inliers')
plt.scatter(X[~ransac.inliner_mask], y[~ransac.inliner_mask], color='red', marker='.', label='Outliers')
plt.plot(line_X, line_y, color='blue', linewidth=2, label='RANSAC')
plt.plot(X_test, y_pred_test, color='red', label='Linear regression')
 
plt.legend(loc='lower right')
plt.xlabel("Input")
plt.ylabel("Response")
plt.show()   


