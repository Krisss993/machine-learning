import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics         import r2_score

arr = np.arange(5,30,2)
arr


boolArr = arr<10
boolArr

arr[boolArr]

arr[arr<20]

arr[arr%3==0]
arr[(arr>10) & (arr<20)]

arr = np.arange(0,24).reshape(4, 6)
arr[0]
arr[0][1]
arr[0][1:3]
arr[0][1:4]
arr[0][:]

arr[1][:]

arr[:2,1]
arr[:2,0:2]

arr[0,1:4]

arr[:2,1]
arr[:,1]

arr[:2,1]

arr[:2,1:3]

arr[:,-1]

arr[:, :-1]


arr=np.arange(0,50).reshape(10,5)
arr

split_level = 0.2
num_rows = arr.shape[0]
num_rows
split_boarder = split_level * num_rows
split_boarder

arr[:round(split_boarder)]
arr[round(split_boarder):]

np.random.shuffle(arr)

X_learn = arr[:]


data = np.arange(500).reshape(100, 5)
data
np.random.shuffle(data)
data
split_level = 0.2
num_rows = data.shape[0]
num_rows
split_boarder = round(split_level * num_rows)
split_boarder

X_learn = data[split_boarder:, :-1]
X_learn
X_test = data[:split_boarder, :-1]
X_test

y_learn = data[split_boarder:, -1]
y_learn

y_test = data[:split_boarder, -1]
y_test


X_train, X_test, y_train, y_test = train_test_split(
        data[:, :-1], data[:, -1], test_size=0.2, shuffle = True)


lr = LinearRegression()
lr.fit(X_train , y_train)
y_pred = lr.predict(X_test)
r2 = r2_score(y_test, y_pred)
scores = []

scores.append(r2)
print('Linear Regression R2: {0:.2f}'.format(r2))

 
