import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('[HR_ML]\Data\HRMLclean.csv')
data= data.set_index('EmployeeID')

X = data.drop("attrit", axis=1) #make X all independent variables
y = data[['attrit']] #make y the dependent variable

#create training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .33, random_state = 67)

#Gridsearch SVC method across parameters
para_grid = {'C':[0.01, 0.1, 1, 10, 100, 1000], 'gamma':[1, 0.1, 0.01, 0.001]}
grid = GridSearchCV(SVC(), para_grid, verbose=3)
grid.fit(X_train, y_train)

print(grid.best_params_) #print best fit parameters

grid_pred = grid.predict(X_test) #make predictions on test data

print(confusion_matrix(y_test, grid_pred)) #print confusion matrix
