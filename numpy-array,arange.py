import numpy as np


a = np.arange(0,20,1)
a
a[0]

a = a.reshape(2,10)
a
a.shape
a[1][4]
a = a.reshape(2,5,2)
a
a.shape
a[0]
a[0][4]
a[0][4][1]
a[0][4][0]

b=np.arange(0,40,2).reshape(4,5)
b

a_list = [2**x for x in range(10)]
a_list
c = np.array(a_list)
c

zero_array = np.zeros(10)
zero_array

one_array = np.ones(10)
one_array

empty_array = np.empty(100)
empty_array

lucky_array = np.full(25,13).reshape(5,5)
lucky_array

lucky_array1 = np.full((5,5),13)
lucky_array1

ran_array = np.random.random(10)
ran_array

linspace_array = np.linspace(100, 200, num=5)
linspace_array

