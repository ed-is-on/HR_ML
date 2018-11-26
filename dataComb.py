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
data.SickLeaveHours.fillna(0, inplace=True) #NaN = didn't use sick leave in 2016
data.TeleHours.fillna(0, inplace=True) #NaN = didn't record telework hours in 2016
data.PrmtLstYr.fillna(0, inplace=True) #NaN = wasn't promoted during 2016
data.AnnualLeaveHrs.fillna(0, inplace=True) #NaN = didn't take leave during 2016

#create attrit column - combo of retirement and separated over 2017
def check_left(row):
    if row['Separated'] == 1:
        return 1
    if row['Retired'] == 1:
        return 1
    else:
        return 0

data['attrit'] = data.apply(lambda row: check_left(row),axis=1)

#performance report indicate not rated as a category
data.LstPfmRt.fillna("NR", inplace=True) #NaN = NR as in not rated - likely more categories in NR but cannot pull out at this time
#Retirement category
data.RetPln.fillna("K", inplace=True) #NaN = K as it is the highest frequency plan by a huge margin. Researching more into the missing data.

#convert date columns
data['ERetDt1'] = pd.to_datetime(data['ERetDt']) #create date/time version
data['RetDt1'] = pd.to_datetime(data['RetDt']) #create date/time version
#calculate days until
data['ERetDayUnt'] = data['ERetDt1'].sub(pd.to_datetime('12/31/2016'), axis=0)/ np.timedelta64(1, 'D') #days until early retirement as of Dec 31 2016
data['RetDayUnt'] = data['RetDt1'].sub(pd.to_datetime('12/31/2016'), axis=0)/ np.timedelta64(1, 'D') #days until retirement as of Dec 31 2016
#fill in NaN with average
#data['ERetDayUnt'].hist()
data.ERetDayUnt.fillna(data['ERetDayUnt'].mean(), inplace=True) #NaN = mean of days
data.RetDayUnt.fillna(data['RetDayUnt'].mean(), inplace=True) #NaN = mean of days

#move categoricals to separate columns
poc = pd.get_dummies(data['POC'])
pp = pd.get_dummies(data['Pay_Plan'], prefix='payPln')
pRpt = pd.get_dummies(data['LstPfmRt'], prefix='perfRev')
rPln = pd.get_dummies(data['RetPln'], prefix='retPln')
#srs = pd.get_dummies(data['Series'], prefix='series') #not sure if the ML will take series as a number var instead of category

#merge into data
data = data.join(poc)
data = data.join(pp)
data = data.join(pRpt)
data = data.join(rPln)
#data = data.join(srs)

#drop columns no longer needed
data.drop("Separated", axis=1, inplace=True)
data.drop("Retired", axis=1, inplace=True)
data.drop("ERetDt", axis=1, inplace=True)
data.drop("RetDt", axis=1, inplace=True)
data.drop("ERetDt1", axis=1, inplace=True)
data.drop("RetDt1", axis=1, inplace=True)
data.drop("POC", axis=1, inplace=True)
data.drop("Pay_Plan", axis=1, inplace=True)
data.drop("LstPfmRt", axis=1, inplace=True)
data.drop("RetPln", axis=1, inplace=True)
#data.drop("Series", axis=1, inplace=True)

#---visualize missing data tool  ---#
#sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')

#export data
#data.to_csv('[HR_ML]\Data\HRMLclean.csv')

#print(data.shape)