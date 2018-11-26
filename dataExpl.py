#python data exploration
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

data = pd.read_csv('[HR_ML]\Data\HRMLclean.csv')

#print(data.columns)

#---some visualizations on the data framework
negative = data[data['attrit'].isin([1])]
positive = data[data['attrit'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['TotTrainInvt'], positive['SickLeaveHours'], s=50, c='b', marker='o', label='Not Attrit')
ax.scatter(negative['TotTrainInvt'], negative['SickLeaveHours'], s=50, c='r', marker='x', label='Attrit')
ax.legend()
ax.set_xlabel('Amount of Training $')
ax.set_ylabel('Sick Leave Hours')

plt.show()