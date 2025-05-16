# -*- coding: utf-8 -*-
"""
Created on Mon May  8 14:49:19 2023

@author: Student
"""
import numpy as np               
import pandas as pd              
import matplotlib.pyplot as plt  
import seaborn as sns           

pd.options.display.float_format = '{:.2f}'.format
from scipy.stats import norm

import warnings                   
warnings.filterwarnings('ignore')
sns.set()

#%%
df = pd.read_csv(r'C:\Users\noshin tasnim khan\Downloads\heart_disease.csv')
#%%
## the dataset

df.head(5)

#%%
## Checking the shape of the dataset, rows, columns values

print(f'\033[94mTotal rows =  {df.shape[0]}')
print(f'\033[94mTotal columns =  {df.shape[1]}')
print(f'\033[94mTotal values = {df.count().sum()}')
print(f'\033[94mTotal missing values = {sum(df.isna().sum())}')

#%%
#%%
df.columns
#%%
df.dtypes
#%%
# Use numpy.unique() to unique values in multiple columns 
column_values = df[['male','education',
                    'currentSmoker','BPMeds','prevalentStroke', 
                    'prevalentHyp', 'diabetes','TenYearCHD']].values
df2 = np.unique(column_values)
print(df2)

#%%
# Find unique values of a column
print('male',df['male'].unique())
print('Education',df['education'].unique())
print('currentSmoker',df['currentSmoker'].unique())
print('BPMeds',df['BPMeds'].unique())
print('prevalentStroke',df['prevalentStroke'].unique())
print('prevalentHyp',df['prevalentHyp'].unique())
print('diabetes',df['diabetes'].unique())
print('TenYearCHD',df['TenYearCHD'].unique())

#%% ---------------------CATEGORICAL VALUES-----------------------------
#%%
categorical = ['male','education',
                    'currentSmoker','BPMeds','prevalentStroke', 
                    'prevalentHyp', 'diabetes','TenYearCHD']

## THIS COLUMNS SHOW ONLY 0,1 (YES,NO)0R 1,2,3,4 ( GRAD,UNDERGRAD,postgraduate'
## 'primaryschool)VALUES. THATS WHY THEY ARE CATEGORICAL

#%%
## check the feature data types, value count etc

df.info()

#%%
#%%

## Determining the null value count by feature

pd.isna(df).sum()[pd.isna(df).sum() > 0]

#%%
## Now we proceed with our original plan to remove null values

## Null values in numerical variables

df.education.fillna(df.education.median(),inplace = True)
df.cigsPerDay.fillna(df.cigsPerDay.median(),inplace = True)
df.BPMeds.fillna(df.BPMeds.median(),inplace = True)
df.totChol.fillna(df.totChol.median(),inplace = True)
df.BMI.fillna(df.BMI.median(),inplace = True)
df.heartRate.fillna(df.heartRate.median(),inplace = True)
df.glucose.fillna(df.glucose.median(),inplace = True)



pd.isna(df).sum()[pd.isna(df).sum() > 0]



#%%
## Before proceeding to EDA, let's check the stastical description of th dataframe

df.describe(include='all').T

#%%  ------Taking sample--------
data1 = df.sample(frac=0.1)
data1.info()
#%% stastical description of sample
data1.describe(include='all').T


#%%------------------------------  E D A----------------------------------
#%%
#                                         histogram

#%% ALL
data1['age'].value_counts()
sns.displot(data1['age'], kde=True)

data1['male'].value_counts()
sns.displot(data1['male'], kde=True)

data1['education'].value_counts()
sns.displot(data1['education'], kde=True)

data1['currentSmoker'].value_counts()
sns.displot(data1['currentSmoker'], kde=True)

data1['cigsPerDay'].value_counts()
sns.displot(data1['cigsPerDay'], kde=True)

data1['BPMeds'].value_counts()
sns.displot(data1['BPMeds'], kde=True)

data1['prevalentStroke'].value_counts()
sns.displot(data1['prevalentStroke'], kde=True)

data1['prevalentHyp'].value_counts()
sns.displot(data1['prevalentHyp'], kde=True)

data1['diabetes'].value_counts()
sns.displot(data1['diabetes'], kde=True)

data1['totChol'].value_counts()
sns.displot(data1['totChol'], kde=True)

data1['sysBP'].value_counts()
sns.displot(data1['sysBP'], kde=True)

data1['diaBP'].value_counts()
sns.displot(data1['diaBP'], kde=True)

data1['BMI'].value_counts()
sns.displot(data1['BMI'], kde=True)

data1['heartRate'].value_counts()
sns.displot(data1['heartRate'], kde=True)

data1['glucose'].value_counts()
sns.displot(data1['glucose'], kde=True)

data1['TenYearCHD'].value_counts()
sns.displot(data1['TenYearCHD'], kde=True)




#%% 
#                                      BOXPLOT

#%% ALL
sns.set(style="whitegrid")
sns.boxplot(x = data1['age'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['male'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['education'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['currentSmoker'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['cigsPerDay'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['BPMeds'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['prevalentStroke'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['prevalentHyp'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['diabetes'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['totChol'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['sysBP'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['diaBP'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['BMI'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['heartRate'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['glucose'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['TenYearCHD'])
plt.show()


#%% -----------Removing outliers-------------------
#%%

columns_to_check = ['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds',
       'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
       'diaBP', 'BMI', 'heartRate', 'glucose', 'TenYearCHD']  
for column in columns_to_check:
 
    Q1 = data1[column].quantile(0.25)
    Q3 = data1[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    data1 = data1[(data1[column] >= lower_bound) & (data1[column] <= upper_bound)]


from scipy.stats import zscore

# Calculate the z-scores for the cleaned data
data1_zscore = data1.apply(zscore)

print(data1_zscore)


#%%-----------------Bar plot for categorical data---------------------
#%%

data1['male'].value_counts().plot(kind='bar',color=['blue','purple'])
plt.title("Column-Male")
plt.xlabel("1=Yes, 0=No")
plt.ylabel("no. of people")

plt.show()


data1['education'].value_counts().plot(kind='bar',color=['blue','purple','green','black'])
plt.title("Column-education")
plt.xlabel("0=PrimarySchoolGrad,1=HighSchool,2=Undergrad,3=PostGrad")
plt.ylabel("no. of people")

plt.show()


data1['currentSmoker'].value_counts().plot(kind='bar',color=['blue','purple'])
plt.title("Column-currentSmoker")
plt.xlabel("1=Yes, 0=No")
plt.ylabel("no. of people")

plt.show()


data1['BPMeds'].value_counts().plot(kind='bar',color=['blue','purple'])
plt.title("Column-BPMeds")
plt.xlabel("1=Yes, 0=No")
plt.ylabel("no. of people")

plt.show()


data1['prevalentStroke'].value_counts().plot(kind='bar',color=['blue','purple'])
plt.title("Column-prevalentStroke")
plt.xlabel("1=Yes, 0=No")
plt.ylabel("no. of people")

plt.show()


data1['prevalentHyp'].value_counts().plot(kind='bar',color=['blue','purple'])
plt.title("Column-prevalentHyp")
plt.xlabel("1=Yes, 0=No")
plt.ylabel("no. of people")

plt.show()


data1['diabetes'].value_counts().plot(kind='bar',color=['blue','purple'])
plt.title("Column-diabetes")
plt.xlabel("1=Yes, 0=No")
plt.ylabel("no. of people")

plt.show()


data1['TenYearCHD'].value_counts().plot(kind='bar',color=['blue','purple'])
plt.title("Column-TenYearCHD")
plt.xlabel("1=Yes, 0=No")
plt.ylabel("no. of people")

plt.show()



#%%
#                                           scatterplot

#%%
## Now let's understand the impact of a combination of two numerical variables on Heart Stroke occurence

#%%
sns.set(color_codes=True)

plt.title('Heart_ stroke by Age & Cigarettes / Day', fontdict={'fontsize':15})
sns.scatterplot(x='age', y='cigsPerDay',hue='heartRate',data = data1)
plt.show()

sns.set(color_codes=True)

plt.title('Heart_ stroke by Age & totChol / Day', fontdict={'fontsize':15})
sns.scatterplot(x='age', y='totChol',hue='heartRate', data = data1)
plt.show()

sns.set(color_codes=True)

plt.title('Heart_ stroke by Age & diaBP / Day', fontdict={'fontsize':15})
sns.scatterplot(x='age', y='diaBP',hue='heartRate', data = data1)
plt.show()

sns.set(color_codes=True)

plt.title('Heart_ stroke by Age & BMI / Day', fontdict={'fontsize':15})
sns.scatterplot(x='age', y='BMI',hue='heartRate', data = data1)
plt.show()


sns.set(color_codes=True)

plt.title('Heart_ stroke by Age & heartRate / Day', fontdict={'fontsize':15})
sns.scatterplot(x='age', y='heartRate',hue='heartRate', data = data1)
plt.show()

sns.set(color_codes=True)

plt.title('Heart_ stroke by Age & glucose / Day', fontdict={'fontsize':15})
sns.scatterplot(x='age', y='glucose',hue='heartRate', data = data1)
plt.show()

sns.set(color_codes=True)

plt.title('Heart_ stroke by totChol & Cigarettes / Day', fontdict={'fontsize':15})
sns.scatterplot(x='totChol', y='cigsPerDay',hue='heartRate', data = data1)
plt.show()

sns.set(color_codes=True)

plt.title('Heart_ stroke by sysBP & diaBP / Day', fontdict={'fontsize':15})
sns.scatterplot(x='sysBP', y='diaBP',hue='heartRate',data = data1)
plt.show()

#%%
#                                    Co-relation

#%%
#%%
plt.figure(figsize=(14,9))
#sns.set_theme(style="white")
corr = data1.corr()
heatmap = sns.heatmap(corr, annot=True, cmap="Blues", fmt='.1g')




#%%
#Finding p value for age
age = df[['age']]

print(age.mean())
#%%

df1=pd.DataFrame(data1,columns=['age'])
print(df1)
#%%
from scipy.stats import ttest_1samp
ttest,p_value=ttest_1samp(df1,49)
print(p_value)
#%%
if p_value<.05:
    print("We are rejecting null hypothesis")
else:
    print("We are accepting null hypothesis")
#%%
#finding p value for sysBP
sysBP = df[['sysBP']]

print(sysBP.mean())
#%%

df2=pd.DataFrame(data1,columns=['sysBP'])
print(df2)
#%%
from scipy.stats import ttest_1samp
ttest,p_value=ttest_1samp(df2,132)
print(p_value)
#%%
if p_value<.05:
    print("We are rejecting null hypothesis")
else:
    print("We are accepting null hypothesis")
#%%
#finding p value for diaBP
diaBP = df[['diaBP']]

print(diaBP.mean())
#%%

df3=pd.DataFrame(data1,columns=['sysBP'])
print(df3)
#%%
from scipy.stats import ttest_1samp
ttest,p_value=ttest_1samp(df3,82)
print(p_value)
#%%
if p_value<.05:
    print("We are rejecting null hypothesis")
else:
    print("We are accepting null hypothesis")
#%%
##finding p value for BMI
BMI = df[['BMI']]

print(BMI.mean())
#%%

df4=pd.DataFrame(data1,columns=['BMI'])
print(df4)
#%%
from scipy.stats import ttest_1samp
ttest,p_value=ttest_1samp(df4,25)
print(p_value)
#%%
if p_value<.05:
    print("We are rejecting null hypothesis")
else:
    print("We are accepting null hypothesis")
#%%
##finding p value for heartRate
heartRate = df[['heartRate']]

print(heartRate.mean())
#%%

df5=pd.DataFrame(data1,columns=['heartRate'])
print(df5)
#%%
from scipy.stats import ttest_1samp
ttest,p_value=ttest_1samp(df5,75)
print(p_value)
#%%
if p_value<.05:
    print("We are rejecting null hypothesis")
else:
    print("We are accepting null hypothesis")
#%%
#%%As there is no catagical variable .so, chi square will not be calculated.
categorical = ['male','education',
                    'currentSmoker','BPMeds','prevalentStroke', 
                    'prevalentHyp', 'diabetes','TenYearCHD']
#%% Chi-square test
#Is male associated with currentSmoker?
             
import scipy.stats as stats
          
dataset=pd.crosstab(df['male'],df['currentSmoker'])
print(dataset)
#%% Observed Values:
Observed_value=dataset.values
print("Observed Values:\n", Observed_value)
#%%
#expected value
val=stats.chi2_contingency(dataset)
print(val)
expected_value=val[3]
expected_value
#%%
no_of_rows=len(dataset.iloc[0:2,0])
no_of_columns=len(dataset.iloc[0,0:2])
ddof=(no_of_rows-1)*(no_of_columns-1)
print("Degree of Freedom:", ddof)
alpha=.05
#%%
from scipy.stats import chi2
chi_square=sum([(o-e)**2./e for o,e in zip(Observed_value,expected_value)])
chi_square_stat=chi_square[0]+chi_square[1]
print("Chi_square statistics:",chi_square_stat)
#%%
critical_value=chi2.ppf(q=1-alpha,df=ddof)
print("Critical value:",critical_value)

#%%
if chi_square_stat>=critical_value:
    print("Reject H0,There is a relationship between 2 categorical varaiables")
else:
    print("Retain H0,There is no relationship between 2 categorical varaiables")
   
#%% Hypotheisis testing for 1 sample

#%%Hypothesis testing for 2 samples

#Is there a significant relationship between #sysBP and diaBp in a given dataset?
#sysBP vs diaBP
import matplotlib
from matplotlib import pyplot
pyplot.figure(figsize=(14,6))
pyplot.hist(df['sysBP'])
pyplot.show()
#%%
import matplotlib
from matplotlib import pyplot
pyplot.figure(figsize=(14,6))
pyplot.hist(df['diaBP'])
pyplot.show()
#%%

df7=pd.DataFrame(data1,columns=['sysBP'])
print(df7)
#%%

df8=pd.DataFrame(data1,columns=['diaBP'])
print(df8)
#%%

print(df7.mean())
print(df8.mean())
#%%One sample T test
import scipy
scipy.stats.ttest_1samp(df7,0)#suppose the null hypothesis is 0
#%%
if p_value<.05:
    print("We are rejecting null hypothesis")
else:
    print("We are accepting null hypothesis")
#%% 2 sample T test Indepependent
#null hypothesis is mean of sysBP and diaBP is equal
scipy.stats.ttest_ind(df7,df8)
#p value is very small.so,we can say that it rejects null hypothesis.


