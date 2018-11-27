import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense


# fix random seed for reproducibility
np.random.seed(5)

#read in data
data = pd.read_csv('[HR_ML]\Data\HRMLclean.csv')
index= data[['EmployeeID']].copy()
data.drop('EmployeeID', axis=1, inplace=True)
data.drop('payPln_AD', axis=1, inplace=True)
data.drop('payPln_AL', axis=1, inplace=True)
data.drop('payPln_EF', axis=1, inplace=True)
data.drop('payPln_ES', axis=1, inplace=True)
data.drop('payPln_EX', axis=1, inplace=True)
data.drop('payPln_GL', axis=1, inplace=True)
data.drop('payPln_GM', axis=1, inplace=True)
data.drop('payPln_GS', axis=1, inplace=True)
data.drop('payPln_SL', axis=1, inplace=True)

#normalize data from 0-1
x = data.values 
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
data_temp = pd.DataFrame(x_scaled, columns=data.columns)

#merge in employeeID and set as index
dataS= index.merge(data_temp, left_index=True, right_index=True, how='outer')
dataS.set_index('EmployeeID', inplace=True)
#print(dataS.head())

#break out dependent variable
X = dataS.drop("attrit", axis=1) #make X all independent variables
y = dataS[['attrit']] #make y the dependent variable

#create training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 67)

# create model
model = Sequential()
model.add(Dense(26, input_dim=45, kernel_initializer='RandomNormal',
                bias_initializer='ones', activation='relu'))
model.add(Dense(90, kernel_initializer='RandomNormal', activation='relu'))
model.add(Dense(45, kernel_initializer='RandomNormal', activation='relu'))
model.add(Dense(5, kernel_initializer='RandomNormal', activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=50, batch_size=10)

# evaluate the model
Xpred = model.predict(X_test)
plt.hist(Xpred, bins=50)
plt.show()
x_pred = (Xpred > 0.5).astype(int) #0.1 reduces type I error to almost 99.2%; 0.45 produces higher accuracy but less control over type I or II error
print(confusion_matrix(y_test, x_pred))
scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#overall accruacy is approximately 91.5-92% but when controlling for type I error it drops into the 80s