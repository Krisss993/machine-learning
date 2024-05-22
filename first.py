import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets._samples_generator import make_blobs


iris = pd.read_csv(r"F:\UdemyMachineLearning\iris\iris.data", header = None, names = ['petal length', 'petal width', 'sepal length', 'sepal width', 'species'])

iris.head()

iris.shape





 
# ... a tutaj podobny wykres generowany przez funkcję pairplot z modułu seaborn

sns.set()
sns.pairplot(iris, hue="species")

pd.plotting.scatter_matrix(iris, figsize=(8, 8))
plt.show()

# x wybiera wszystkie wiersze i kolumny do 4
X = iris.iloc[:, :4]
# y wybiera wszystkie wiersze i kolumne species (5)
y = iris.loc[:, 'species']

categories = {'Iris-setosa':1, 'Iris-versicolor':2, 'Iris-virginica':3}
y = y.apply(lambda x:categories[X])

X.head()
y.head()

lr  = LinearRegression()
lr.fit(X,y)
lr.score(X, y)

iris_1 = [5, 3.5, 1.4, 0.2]
iris_2 = [6.4, 3, 4.5, 1]
iris_3 = [6, 3, 5, 2]
other = [1, 2, 3, 4]
flowers = [iris_1, iris_2, iris_3, other]


species_predict = lr.predict(flowers)

print(species_predict)


for flow, spec in zip(flowers, species_predict):
    if round(spec) == 1:
        print('Flower {} is {}'.format(flow,'Iris-setosa'))
    elif round(spec) == 2:
        print('Flower {} is {}'.format(flow,'Iris-versicolor'))
    elif round(spec) == 3:
        print('Flower {} is {}'.format(flow,'Iris-virginica'))
    else:
        print('Flower {} is {}'.format(flow,'other'))
        


mpg = pd.read_csv(r"F:\UdemyMachineLearning\auto-mpg\auto-mpg.csv")

mpg.head()

mpg.shape

X = mpg.iloc[:, 1:8]
X.drop(axis = 1, columns='horsepower', inplace = True)
X.head()


y = mpg.loc[:, 'mpg']
y.head()



lr = LinearRegression()
lr.fit(X.to_numpy(), y)
lr.score(X.to_numpy(), y)

my_car1 = [4, 160, 190, 12, 90, 1]
my_car2 = [4, 200, 260, 15, 83, 1]
 
cars = [my_car1, my_car2]


mpg_cons_predict = lr.predict(cars)

print(mpg_cons_predict)











# UNSUPERVISED LEARNING





df = pd.read_csv(r"F:\UdemyMachineLearning\Airbnb+listings+in+Ottawa+(May+2016)\Airbnb listings in Ottawa (May 2016).csv")
df.shape
df.head()



coordinates = df.loc[:,['longitude','latitude']]
coordinates.shape

plt.scatter(df.loc[:,'longitude'], df.loc[:, 'latitude'])

WCSS = []

for k in range(1,15):
    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(coordinates)
    WCSS.append(kmeans.inertia_)


plt.plot(range(1,15), WCSS)
plt.xlabel('Nr of K val (cluster')
plt.ylabel('WCSS')
plt.grid()
plt.show()



kmeans = KMeans(n_clusters=4, max_iter=300, random_state=1, n_init='auto')
clusters = kmeans.fit_predict(coordinates)
labels=kmeans.labels_
centroids = kmeans.cluster_centers_


h = 0.001

x_min, x_max = coordinates['longitude'].min(), coordinates['longitude'].max()
y_min, y_max = coordinates['latitude'].min(), coordinates['latitude'].max()


xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h ))

Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])



# WYKRES WPŁYWÓG/GRANIC
plt.figure(1, figsize=(10,4))
plt.clf()
Z = Z.reshape(xx.shape)
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap= plt.cm.Pastel1, origin='lower')

# DODANE DO WYKRESU ODPOWIEDNICH PUNKTÓW x, y ORAZ centroid
plt.scatter(x = coordinates['longitude'], y= coordinates['latitude'], c=labels, s=100)
plt.scatter(x= centroids[:,0], y=centroids[:,1], s=300, c='red')

plt.ylabel('long(y')
plt.xlabel('lat(x)')
plt.grid()
plt.title('Clustering')
plt.show()

################################




# WYKRES ROZŁOŻENIA PUNKTÓW

X, y = make_blobs(n_samples=100, centers=4, cluster_std=0.6, random_state=0)

plt.scatter(X[:,0], X[:,1])

# WYKRES ŁOKCIA

WCSS = []

for k in range(1,15):
    kwmeans = KMeans(n_clusters=k, n_init='auto')
    kwmeans.fit(X)
    WCSS.append(kwmeans.inertia_)
    

plt.plot(range(1,15), WCSS)
plt.xlabel('Nr of K val (cluster')
plt.ylabel('WCSS')
plt.grid()
plt.show()

#################################################################################


kmeans = KMeans(n_clusters=4, max_iter=300, random_state=0, n_init='auto')
clusters =kmeans.fit_predict(X)
labels=kmeans.labels_
centroids = kmeans.cluster_centers_



h = 0.01
x_min, x_max = X[:,0].min(), X[:,0].max()
y_min, y_max = X[:,1].min(), X[:,1].max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()]) 
Z = Z.reshape(xx.shape)
 
plt.figure(1 , figsize = (15 , 7) )
plt.clf()
 
plt.imshow(Z , interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel1, origin='lower')
 
plt.scatter(x=X[:,0], y=X[:,1], c=labels, s=100)
 
plt.scatter(x=centroids[:,0], y=centroids[:,1],s=300 , c='red')
 
plt.ylabel('x') , plt.xlabel('y')
plt.grid()
plt.title("Clustering")
plt.show()


