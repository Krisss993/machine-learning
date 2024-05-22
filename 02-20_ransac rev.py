import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.model_selection import train_test_split

class RANSAC1:
    def __init__(self, max_iters=100, treshold=4, min_acceptable_inliers=100):
        self.max_iters=max_iters
        self.treshold=treshold
        self.min_acceptable_inliers=min_acceptable_inliers
        
        self.best_model=None
        self.best_inliers_count=0
        self.inliers_mask=None
        
    def fit(self, X, y, show_partial_res=False):
        assert X.shape[1] == 1, '1'
        assert X.shape[0] > 1.5 * self.min_acceptable_inliers, 'G'
        
        sample_size = X.shape[1] + 1
        data = np.hstack((X[:,0].reshape(-1,1), y.reshape(-1,1)))
        
        for i in range(self.max_iters):
            idx_rand = np.random.choice(len(data), size=sample_size, replace=False)
            points = data[idx_rand]
            
            a=(points[0,1]-points[1,1])/(points[0,0]-points[1,0])
            b=points[0,1]-a*points[0,0]
            
            y_pred = a*data[:,0] + b
            
            this_inlier_mask = np.square(y-y_pred) < self.treshold
            this_inlier_count = np.sum(this_inlier_mask)
            
            better_choice = (this_inlier_count > self.best_inliers_count) and (this_inlier_count >self.min_acceptable_inliers)
            if better_choice:
                self.best_model=(a,b)
                self.best_inliers_count=this_inlier_count
                self.inliers_mask=this_inlier_mask
                
            if show_partial_res:
                line_X = np.arange(X.min(), X.max()).reshape(-1,1)
                line_y = a * line_X + b
                plt.scatter(X[this_inlier_mask], y[this_inlier_mask], color='green')
                plt.scatter(X[~this_inlier_mask], y[~this_inlier_mask], color='red')
                plt.plot(line_X, line_y)
                plt.scatter(points[:,0], points[:,1])
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

ran=RANSAC1(max_iters=100, treshold=6, min_acceptable_inliers=100)

ran.fit(X_train, y_train, show_partial_res=True)

