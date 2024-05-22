import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LinearRegression


animals = pd.read_csv(r'F:/\UdemyMachineLearning/animals/animals.csv')
animals
animals = animals[animals['name'].isin(['Cow','Goat','Donkey','Horse','Giraffe','Kangaroo','Rabbit','Sheep','Mole','Pig'])]
animals


plt.figure(figsize=(7,5))
plt.scatter(animals['body'],animals['brain'], color='blue')
plt.show()

lr = LinearRegression()
lr.fit(X = animals['body'].values.reshape(-1,1), y = animals['brain'].values)

brain_pred = lr.predict(X = animals['body'].values.reshape(-1,1))


plt.figure(figsize=(7,5))
plt.scatter(animals['body'],animals['brain'], color='blue')
plt.plot(animals['body'], brain_pred, color='red', linewidth=2)
plt.show()

new_animals_body = np.array([100,200,300,400])
new_animals_brain = lr.predict(new_animals_body.reshape(-1,1))


plt.figure(figsize=(7,5))
plt.scatter(animals['body'],animals['brain'], color='blue')
plt.plot(animals['body'], brain_pred, color='red', linewidth=2)
plt.scatter(new_animals_body,new_animals_brain, color='black', s=100)
plt.show()

print(lr.coef_)
print(lr.intercept_)





###############################################################################






data = pd.read_csv(r"F:/\UdemyMachineLearning/high_school_sat_gpa/high_school_sat_gpa.csv", sep=' ', usecols=['math_SAT','verb_SAT','high_GPA'])
data
data.dtypes

plt.scatter(data['math_SAT'], data['high_GPA'], color='blue')

lr=LinearRegression()
lr.fit(X = data['math_SAT'].values.reshape(-1,1), y = data['high_GPA'].values)


pred = lr.predict(X = data['math_SAT'].values.reshape(-1,1))

x_min = min(data['math_SAT'])
x_min

x_max = max(data['math_SAT'])
x_max

plt.figure(figsize=(7,5))
plt.scatter(data['math_SAT'], data['high_GPA'], color='blue')
plt.plot([x_min,x_max], lr.predict([[x_min],[x_max]]), color='red')
plt.show()


plt.figure(figsize=(7,5))
plt.scatter(data['verb_SAT'],data['high_GPA'], color = 'black')
plt.show()
lr_verb = LinearRegression()
lr_verb.fit(X = data['verb_SAT'].values.reshape(-1,1), y = data['high_GPA'].values)

x_min = min(data['verb_SAT'])
x_max = max(data['verb_SAT'])

plt.figure(figsize=(7,5))
plt.scatter(data['verb_SAT'],data['high_GPA'], color = 'black')
plt.plot([x_min,x_max], lr_verb.predict([[x_min], [x_max]]), color='red')
plt.show()

lr2 = LinearRegression()

lr2.fit(X = data[['math_SAT','verb_SAT']].values, y = data['high_GPA'].values.reshape(-1,1))

john = np.array([600,650]).reshape(1,2)

lr2.predict(john)
