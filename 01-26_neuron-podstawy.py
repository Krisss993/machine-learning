import time
import numpy as np
import matplotlib.pyplot as plt
 
 
num_iterations1 = 30
time_results_loop = []
 
for iteration in range(1, num_iterations1+1):
    
    start_time = time.time()
    
    data1 = np.ones(shape=(10*iteration, 10*iteration), dtype=np.float64)
    data2 = np.ones(shape=(10*iteration, 10*iteration), dtype=np.float64)
    data3 = np.zeros(shape=(10*iteration, 10*iteration), dtype=np.float64)
    
    data3 = data1.dot(data2)
    
    end_time = time.time()
    
    print('{} - :{}'.format(iteration, end_time - start_time))    
    time_results_loop.append(end_time - start_time)


num_iterations2 = 100
time_results_np = []
 
for iteration in range(1, num_iterations2+1):
 
    start_time = time.time()
    
    data = np.arange(0,10000*iteration, 1)
    my_sum = np.sum(data)
    
    end_time = time.time()
    
    print('{} - :{}'.format(iteration, end_time - start_time))    
    time_results_np.append(end_time - start_time)
    
    
    
fig = plt.figure()
plt.scatter(range(num_iterations1), time_results_loop, s=10, c='b', marker="s", label='loop')
plt.scatter(range(num_iterations2), time_results_np, s=10, c='r', marker="o", label='numpy')
plt.legend(loc='upper left');
plt.show()

