import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, eta=0.10, epochs=50, is_verbose = False):
        self.eta=eta
        self.epochs=epochs
        self.is_verbose=is_verbose
        self.list_of_errors=[]
        
    # x to wektor danych (zbiór cech jednej obserwacji)
    # z to total_stimulation
    def predict(self, x):
        total_stimulation = np.dot(x, self.w)
        #print('Z: ',total_stimulation)
        y_pred = 1 if total_stimulation > 0 else -1
        return y_pred
    
    # y to etykiety danych
    def fit(self, X, y):
        
        self.list_of_errors = []
        
        ones = np.ones((X.shape[0],1))
        X_1 = np.append(X.copy(), ones, axis=1)
        
        self.w = np.array([0.38544573,0.40046814,0.50987714,0.94058191])
        
        print('w:               ',self.w)
        for e in range(self.epochs):
            
            nr_of_errors = 0
            
            for x, y_target in zip(X_1, y):
                y_pred = self.predict(x)
                delta_w = self.eta * (y_target - y_pred) * x 
                
                self.w += delta_w
                
                nr_of_errors +=1 if y_target != y_pred else 0
                
            self.list_of_errors.append(nr_of_errors)
            
            if (self.is_verbose):
                print('Epoch: {}, weights: {}, number of errors {}'.format(e, self.w, nr_of_errors))
        
                
X = np.array([
    [2,4,20],
    [4,3,-10],
    [5,6,13],
    [5,4,8],
    [3,4,5],
])


y = np.array([1,-1,-1,1,-1])

perc = Perceptron(epochs=100, is_verbose=True)
perc.fit(X, y)
print(perc.w)

#perc.predict(np.array([1,2,3,1]))
#perc.predict(np.array([2,2,8,1]))
#perc.predict(np.array([3,3,3,1]))
print(perc.w)

