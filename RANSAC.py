import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LinearRegression
from scipy import stats

rand_a = np.random.rand(2,5)
rand_b = np.random.rand(1,5)*100

rand_c = np.vstack((rand_a, rand_b))
rand_c

rand_b = np.random.rand(2,1)

rand_c = np.hstack((rand_a, rand_b))
rand_c

X = rand_c[:,2]
X

a,b=3,8

y=a*X+b
y

idx = np.random.choice(10,size=3,replace=False)
idx

X = np.random.randint(low=0, high=10, size=10)
X = X*10
X

print(X[idx])
