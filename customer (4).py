# -*- coding: utf-8 -*-
"""
Created on Wed May 10 00:07:36 2023

@author: Lenovo
"""

import numpy as np              
import pandas as pd              
import matplotlib.pyplot as plt  
import seaborn as sns            
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.2f}'.format
from scipy.stats import norm

import warnings                   
warnings.filterwarnings('ignore')
sns.set()

#%%
df = pd.read_csv(r'C:\Users\Lenovo\Downloads\great_customers.csv')
#%%
## Have a glance at the dataset

df.head(5)

#%%
print(f'\033[94mTotal missing values = {sum(df.isna().sum())}')

#%%
## Checking the shape of the dataset, rows, columns, & missing values

print(f'\033[94mTotal rows =  {df.shape[0]}')
print(f'\033[94mTotal columns =  {df.shape[1]}')
print(f'\033[94mTotal values = {df.count().sum()}')

#%%
print(f'\033[94mTotal missing values = {sum(df.isna().sum())}')

#%%
#%%
df.columns
#%%
df.dtypes
#%%
# Use numpy.unique() to unique values in multiple columns 
column_values = df[['great_customer_class']].values
df2 = np.unique(column_values)
print(df2)

#%%
# Find unique values of a column
print('workclass',df['great_customer_class'].unique())

#%% ---------------------CATEGORICAL VALUES-----------------------------
#%%
categorical = ['great_customer_class']
print(categorical)

## THIS COLUMNS SHOW ONLY 0,1 (YES,NO)0R 1,2,3,4 ( GRAD,UNDERGRAD,postgraduate'
## 'primaryschool)VALUES. THATS WHY THEY ARE CATEGORICAL

#%%
## Gauge the dataframe to check the feature data types, value count, memory usage etc

df.info()

#%%
## Determining the null value count by feature

pd.isna(df).sum()[pd.isna(df).sum() > 0]

#%%
## Now we proceed with our original plan to remove null values

## Null values in numerical variables

df.age.fillna(df.age.median(),inplace = True)
df.salary.fillna(df.salary.median(),inplace = True)
df.mins_beerdrinking_year.fillna(df.mins_beerdrinking_year.median(),inplace = True)
df.mins_exercising_year.fillna(df.mins_exercising_year.median(),inplace = True)
df.tea_per_year.fillna(df.tea_per_year.median(),inplace = True)
df.coffee_per_year.fillna(df.coffee_per_year.median(),inplace = True)


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
data1['user_id'].value_counts()
sns.displot(data1['user_id'], kde=True)

data1['age'].value_counts()
sns.displot(data1['age'], kde=True)

data1['salary'].value_counts()
sns.displot(data1['salary'], kde=True)

data1['education_rank'].value_counts()
sns.displot(data1['education_rank'], kde=True)

data1['mins_beerdrinking_year'].value_counts()
sns.displot(data1['mins_beerdrinking_year'], kde=True)

data1['mins_exercising_year'].value_counts()
sns.displot(data1['mins_exercising_year'], kde=True)

data1['works_hours'].value_counts()
sns.displot(data1['works_hours'], kde=True)

data1['tea_per_year'].value_counts()
sns.displot(data1['tea_per_year'], kde=True)

data1['coffee_per_year'].value_counts()
sns.displot(data1['coffee_per_year'], kde=True)

data1['great_customer_class'].value_counts()
sns.displot(data1['great_customer_class'], kde=True)

#%% 
#                                      BOXPLOT

#%% ALL
sns.set(style="whitegrid")
sns.boxplot(x = data1['user_id'])

sns.set(style="whitegrid")
sns.boxplot(x = data1['age'])

sns.set(style="whitegrid")
sns.boxplot(x = data1['salary'])

sns.set(style="whitegrid")
sns.boxplot(x = data1['education_rank'])

sns.set(style="whitegrid")
sns.boxplot(x = data1['mins_beerdrinking_year'])

sns.set(style="whitegrid")
sns.boxplot(x = data1['mins_exercising_year'])

sns.set(style="whitegrid")
sns.boxplot(x = data1['works_hours'])

sns.set(style="whitegrid")
sns.boxplot(x = data1['tea_per_year'])

sns.set(style="whitegrid")
sns.boxplot(x = data1['coffee_per_year'])

sns.set(style="whitegrid")
sns.boxplot(x = data1['great_customer_class'])

#%%
#*******************************Bar*****************************

#%% 
plt.figure(figsize=(8,10))
sns.countplot(x='workclass',data=data1, palette='rainbow')
plt.title("workclass")
plt.show()

plt.figure(figsize=(8,10))
sns.countplot(x='marital-status',data=data1, palette='rainbow')
plt.title("marital-status")
plt.show()

plt.figure(figsize=(10,8))
sns.countplot(x='occupation',data=data1, palette='rainbow')
plt.title("occupation  ")
plt.show()


plt.figure(figsize=(8,5))
sns.countplot(x='race',data=data1, palette='rainbow')
plt.title("race ")
plt.show()


plt.figure(figsize=(8,5))
sns.countplot(x='sex',data=data1, palette='rainbow')
plt.title("sex")
plt.show()

#%%
#                                           scatterplot

#%%
## Now let's understand the impact of a combination of two numerical variables on Heart Stroke occurence
#sns.scatterplot(x='day', y='tip', data=tip, hue='time')
#%%
sns.set(color_codes=True)

plt.title('Great_Costomer by Age & salary / Day', fontdict={'fontsize':15})
sns.scatterplot(x='age', y='salary',hue='education_rank',data = data1)
plt.show()

sns.set(color_codes=True)

plt.title('Great_Costomer by Age & education_rank / Day', fontdict={'fontsize':15})
sns.scatterplot(x='age', y='education_rank' ,hue='salary', data = data1)
plt.show()

sns.set(color_codes=True)

plt.title('Great_Costomer by Age & works_hours / Day', fontdict={'fontsize':15})
sns.scatterplot(x='age', y='works_hours',hue='salary', data = data1)
plt.show()

sns.set(color_codes=True)

plt.title('Great_Costomer by coffe_per_year & tea_per_year / Day', fontdict={'fontsize':15})
sns.scatterplot(x='age',y='tea_per_year',hue='salary',data = data1)
plt.show()

sns.set(color_codes=True)

plt.title('Great_Costomer by Age & coffee_per_year / Day', fontdict={'fontsize':15})
sns.scatterplot(x='age', y='coffee_per_year',hue='salary',data = data1)
plt.show()

sns.set(color_codes=True)


#%% -----------Removing outliers-------------------
#%%

columns_to_check = ['user_id', 'age', 'workclass', 'salary', 'education_rank',
       'marital-status', 'occupation', 'race', 'sex', 'mins_beerdrinking_year',
       'mins_exercising_year', 'works_hours', 'tea_per_year',
       'coffee_per_year', 'great_customer_class']

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



#%%
#                                    Co-relation

#%%
#%%
plt.figure(figsize=(14,9))
#sns.set_theme(style="white")
corr = data1.corr()
heatmap = sns.heatmap(corr, annot=True, cmap="Blues", fmt='.1g')
#%%
#finding p value for age
age= df[['age']]

print(age.mean())
#%%

df1=pd.DataFrame(data1,columns=['age'])
print(df1)
#%%
from scipy.stats import ttest_1samp
ttest,p_value=ttest_1samp(df1,45)
print(p_value)
#%%
if p_value<.05:
    print("We are rejecting null hypothesis")
else:
    print("We are accepting null hypothesis")
#%%
#finding p value for education_rank

education_rank= df[['education_rank']]

print(education_rank.mean())
#%%

df2=pd.DataFrame(data1,columns=['education_rank'])
print(df2)
#%%
from scipy.stats import ttest_1samp
ttest,p_value=ttest_1samp(df2,10)
print(p_value)
#%%
if p_value<.05:
    print("We are rejecting null hypothesis")
else:
    print("We are accepting null hypothesis")
#%%
#finding p value for works_hours


works_hours= df[['works_hours']]

print(works_hours.mean())
#%%

df3=pd.DataFrame(data1,columns=['works_hours'])
print(df3)
#%%
from scipy.stats import ttest_1samp
ttest,p_value=ttest_1samp(df3,40)
print(p_value)
#%%
if p_value<.05:
    print("We are rejecting null hypothesis")
else:
    print("We are accepting null hypothesis")
#%%
#%% Chi-square test
#Is sex associated with work class?
             
import scipy.stats as stats
          
dataset=pd.crosstab(df['sex'],df['workclass'])
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
no_of_rows=len(dataset.iloc[0:3,0])
no_of_columns=len(dataset.iloc[0,0:3])
ddof=(no_of_rows-1)*(no_of_columns-1)
print("Degree of Freedom:", ddof)
alpha=.05
#%%
from scipy.stats import chi2
chi_square=sum([(o-e)**2/e for o,e in zip(Observed_value,expected_value)])
chi_square_stat=chi_square[0]+chi_square[1]+chi_square[2]
print("Chi_square statistics:",chi_square_stat)
#%%
critical_value=chi2.ppf(q=1-alpha,df=ddof)
print("Critical value:",critical_value)
#%%
if chi_square_stat>=critical_value:
    print("Reject H0,There is a relationship between 2 categorical varaiables")
else:
    print("Retain H0,There is no relationship between 2 categorical varaiables")
#%%
#%% Hypotheisis testing for 1 sample

#%%Hypothesis testing for 2 samples
#Is there a significant difference in the average consumption of tea and coffee per year in a given population?


import matplotlib
from matplotlib import pyplot
pyplot.figure(figsize=(14,6))
pyplot.hist(df['tea_per_year'])
pyplot.show()
#%%
import matplotlib
from matplotlib import pyplot
pyplot.figure(figsize=(14,6))
pyplot.hist(df['coffee_per_year'])
pyplot.show()
#%%

df7=pd.DataFrame(data1,columns=['tea_per_year'])
print(df7)
#%%

df8=pd.DataFrame(data1,columns=['coffee_per_year'])
print(df8)
#%%

print(df7.mean())
print(df8.mean())

#%% 2 sample T test Indepependent

scipy.stats.ttest_ind(df7,df8)
#p value is very small.so,we can say that it rejects null hypothesis.






















