import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 




#read in data files
data = pd.read_csv('[HR_ML]\Data\AggBaseData.csv')
perf = pd.read_csv('[HR_ML]\Data\AggPfrmLst2016.csv')
prom = pd.read_csv('[HR_ML]\Data\AggPrmt2016.csv')
leav = pd.read_csv('[HR_ML]\Data\AggregateLeave2016.csv')
retElg = pd.read_csv('[HR_ML]\Data\AggRet2016.csv')
retd = pd.read_csv('[HR_ML]\Data\AggRetired.csv')
separ = pd.read_csv('[HR_ML]\Data\AggSeparated.csv')
sick = pd.read_csv('[HR_ML]\Data\AggSickLeave2016.csv')
tele = pd.read_csv('[HR_ML]\Data\AggTele.csv')
train = pd.read_csv('[HR_ML]\Data\AggTrainInv.csv')

#set EmployeeID to index
data = data.set_index('EmployeeID')
perf = perf.set_index('EmployeeID')
prom = prom.set_index('EmployeeID')
leav = leav.set_index('EmployeeID')
retElg = retElg.set_index('EmployeeID')
retd = retd.set_index('EmployeeID')
separ = separ.set_index('EmployeeID')
sick = sick.set_index('EmployeeID')
tele = tele.set_index('EmployeeID')
train = train.set_index('EmployeeID')

#join into core dataset
data= data.join(perf)
data = data.join(prom)
data = data.join(leav)
data = data.join(retElg)
data = data.join(retd)
data = data.join(separ)
data = data.join(sick)
data = data.join(tele)
data = data.join(train)

###### ------DATA CLEAN UP ------######

#remove unwanted column
data.drop("Started_Training", axis=1, inplace=True)

#replace NaNs
data.TotTrainInvt.fillna(0, inplace=True) #NaN = haven't filled out an SF182 in 2016 -therefore assumed to have had $0 spent on them in training
data.NumTraining.fillna(0, inplace=True) #NaN = haven't filled out an SF182 in 2016 -therefore assumed took 0 training events
data.TotCreditHours.fillna(0, inplace=True) #NaN = haven't filled out an SF182 in 2016 -officially earned 0 credit hours paid by ED 

#create attrit column - combo of retirement and separated over 2017
def check_left(row):
    if row['Separated'] == 1:
        return 1
    if row['Retired'] == 1:
        return 1
    else:
        return 0

data['attrit'] = data.apply(lambda row: check_left(row),axis=1)

#fill in 0's for separated and retired


#print(data.head())
#sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
#plt.show()
#print(data.shape)
#print(data.head())