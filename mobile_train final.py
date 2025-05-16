# -*- coding: utf-8 -*-
"""
Created on Sun May  7 22:04:06 2023

@author: User
"""



import numpy as np               # linear algebra
import pandas as pd              # data processing, dataset file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # data visualization & graphical plotting
import seaborn as sns            # to visualize random distributions
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.2f}'.format
from scipy.stats import norm

import warnings                   # to deal with warning messages
warnings.filterwarnings('ignore')
sns.set()

#%%
df = pd.read_csv(r'C:\Users\noshin tasnim khan\Downloads\mobile_price_train.csv')

#%%
## Have a glance at the dataset

df.head(5)

#%%
## Checking the shape of the dataset, rows, columns, & missing values

print(f'\033[94mTotal rows =  {df.shape[0]}')
print(f'\033[94mTotal columns =  {df.shape[1]}')
print(f'\033[94mTotal values = {df.count().sum()}')
print(f'\033[94mTotal missing values = {sum(df.isna().sum())}')


#%%
df.columns
#%%
df.dtypes
#%%
# Use numpy.unique() to unique values in multiple columns 
column_values = df[['blue' , 'dual_sim' , 'four_g' , 
                    'n_cores' , 'three_g' , 'touch_screen' ,'wifi' , 
                    'price_range']].values
df2 = np.unique(column_values)
print(df2)

#%%
# Find unique values of a column
print('blue',df['blue'].unique())
print('dual_sim',df['dual_sim'].unique())
print('four_g',df['four_g'].unique())
print('n_cores',df['n_cores'].unique())
print('three_g' ,df['three_g' ].unique())
print('touch_screen',df['touch_screen'].unique())
print('wifi',df['wifi'].unique())
print('price_range',df['price_range'].unique())

#%% ---------------------CATEGORICAL VALUES-----------------------------
#%%
categorical = ['blue' , 'dual_sim' , 'four_g' , 
                    'n_cores' , 'three_g' , 'touch_screen' ,'wifi' , 
                    'price_range']

## THIS COLUMNS SHOW ONLY 0,1 (YES,NO)0R 1,2,3,4 ( GRAD,UNDERGRAD,postgraduate'
## 'primaryschool)VALUES. THATS WHY THEY ARE CATEGORICAL

#%%
## Gauge the dataframe to check the feature data types, value count, memory usage etc

df.info()

#%%
## Determining the null value count by feature

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
data1['battery_power'].value_counts()
sns.displot(data1['battery_power'],kind="hist", kde=True)

data1['blue'].value_counts()
sns.displot(data1['blue'], kde=True)

data1['clock_speed'].value_counts()
sns.displot(data1['clock_speed'], kde=True)

data1['dual_sim'].value_counts()
sns.displot(data1['dual_sim'], kde=True)

data1['fc'].value_counts()
sns.displot(data1['fc'], kde=True)

data1['four_g'].value_counts()
sns.displot(data1['four_g'], kde=True)

data1['int_memory'].value_counts()
sns.displot(data1['int_memory'], kde=True)

data1['m_dep'].value_counts()
sns.displot(data1['m_dep'], kde=True)

data1['mobile_wt'].value_counts()
sns.displot(data1['mobile_wt'], kde=True)

data1['n_cores'].value_counts()
sns.displot(data1['n_cores'], kde=True)

data1['pc'].value_counts()
sns.displot(data1['pc'], kde=True)

data1['px_height'].value_counts()
sns.displot(data1['px_height'], kde=True)

data1['px_width'].value_counts()
sns.displot(data1['px_width'], kde=True)

data1['ram'].value_counts()
sns.displot(data1['ram'], kde=True)

data1['sc_h'].value_counts()
sns.displot(data1['sc_h'], kde=True)

data1['talk_time'].value_counts()
sns.displot(data1['talk_time'], kde=True)

data1['three_g'].value_counts()
sns.displot(data1['three_g'], kde=True)

data1['touch_screen'].value_counts()
sns.displot(data1['touch_screen'], kde=True)

data1['wifi'].value_counts()
sns.displot(data1['wifi'], kde=True)

data1['price_range'].value_counts()
sns.displot(data1['price_range'], kde=True)





#%% 
#                                      BOXPLOT

#%% ALL
sns.set(style="whitegrid")
sns.boxplot(x = data1['battery_power'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['blue'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['clock_speed'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['dual_sim'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['fc'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['four_g'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['int_memory'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['m_dep'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['mobile_wt'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['n_cores'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['pc'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['px_height'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['px_width'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['ram'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['sc_h'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['talk_time'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['three_g'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['touch_screen'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['wifi'])
plt.show()

sns.set(style="whitegrid")
sns.boxplot(x = data1['price_range'])
plt.show()


#%% -----------Removing outliers-------------------
#%%

columns_to_check = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
       'touch_screen', 'wifi', 'price_range']  
for column in columns_to_check:
  
    Q1 = data1[column].quantile(0.25)
    Q3 = data1[column].quantile(0.75)
    IQR = Q3 - Q1

   
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter the data to keep only the values within the bounds
    data1 = data1[(data1[column] >= lower_bound) & (data1[column] <= upper_bound)]


from scipy.stats import zscore

# Calculate the z-scores for the cleaned data
data1_zscore = data1.apply(zscore)

print(data1_zscore)


#%%-----------------Bar plot for categorical data---------------------
#%%

data1['blue'].value_counts().plot(kind='bar',color=['red','green'])
plt.title("Column-blue")
plt.xlabel("1=Yes, 0=No")
plt.ylabel("no. of devices")

plt.show()


data1['dual_sim'].value_counts().plot(kind='bar',color=['blue','purple','green','black'])
plt.title("Column-dual_sim")
plt.xlabel("1=Yes, 0=No")
plt.ylabel("no. of devices")

plt.show()


data1['four_g'].value_counts().plot(kind='bar',color=['blue','purple'])
plt.title("Column-four_g")
plt.xlabel("1=Yes, 0=No")
plt.ylabel("no. of devices")

plt.show()


data1['n_cores'].value_counts().plot(kind='bar',color=['blue'])
plt.title("Column-n_cores")
plt.xlabel("no of cores")
plt.ylabel("no. of devices")

plt.show()


data1['three_g'].value_counts().plot(kind='bar',color=['blue','purple'])
plt.title("Column-three_g")
plt.xlabel("1=Yes, 0=No")
plt.ylabel("no. of devices")

plt.show()


data1['touch_screen'].value_counts().plot(kind='bar',color=['blue','purple'])
plt.title("Column-touch_screen")
plt.xlabel("1=Yes, 0=No")
plt.ylabel("no. of devices")

plt.show()


data1['wifi'].value_counts().plot(kind='bar',color=['blue','purple'])
plt.title("Column-wifi")
plt.xlabel("1=Yes, 0=No")
plt.ylabel("no. of devices")

plt.show()


data1['price_range'].value_counts().plot(kind='bar',color=['blue','purple','green','orange'])
plt.title("Column-price_range")
plt.xlabel("0=VeryHigh, 1=High, 2= Medium, 3=Low ")
plt.ylabel("no. of devices")

plt.show()

#%%
#                                           scatterplot

#%%
## the impact of a combination of two numerical variables on Price Range

#%%
sns.set(color_codes=True)

plt.title('Price_range by Ram and Battery power', fontdict={'fontsize':15})
sns.scatterplot(x='ram', y='battery_power',hue='price_range',data = data1)
plt.show()


sns.set(color_codes=True)
plt.title('Price_range by Ram and color Blue', fontdict={'fontsize':15})
sns.scatterplot(x='ram', y='blue',hue='price_range',data = data1)
plt.show()

sns.set(color_codes=True)
plt.title('Price_range by Ram and Clock Speed', fontdict={'fontsize':15})
sns.scatterplot(x='ram', y='clock_speed',hue='price_range',data = data1)
plt.show()

sns.set(color_codes=True)
plt.title('Price_range by Ram and Dual Sim', fontdict={'fontsize':15})
sns.scatterplot(x='ram', y='dual_sim',hue='price_range',data = data1)
plt.show()

sns.set(color_codes=True)
plt.title('Price_range by Ram and fc', fontdict={'fontsize':15})
sns.scatterplot(x='ram', y='fc',hue='price_range',data = data1,color='green')
plt.show()

sns.set(color_codes=True)
plt.title('Price_range by Ram and Four_g', fontdict={'fontsize':15})
sns.scatterplot(x='ram', y='four_g',hue='price_range',data = data1)
plt.show()


sns.set(color_codes=True)
plt.title('Price_range by Ram and Int_memory', fontdict={'fontsize':15})
sns.scatterplot(x='ram', y='int_memory',hue='price_range',data = data1)
plt.show()

sns.set(color_codes=True)
plt.title('Price_range by Ram and M_dep', fontdict={'fontsize':15})
sns.scatterplot(x='ram',y='m_dep',hue='price_range',data = data1)
plt.show()

sns.set(color_codes=True)
plt.title('Price_range by Ram and Mobile_wt', fontdict={'fontsize':15})
sns.scatterplot(x='ram',y='mobile_wt',hue='price_range',data = data1)
plt.show()

sns.set(color_codes=True)
plt.title('Price_range by Ram and N_crores', fontdict={'fontsize':15})
sns.scatterplot(x='ram',y='n_cores',hue='price_range',data = data1)
plt.show()

sns.set(color_codes=True)
plt.title('Price_range by Ram and Pc', fontdict={'fontsize':15})
sns.scatterplot(x='ram',y='pc',hue='price_range',data = data1)
plt.show()

sns.set(color_codes=True)
plt.title('Price_range by Ram and Px_height', fontdict={'fontsize':15})
sns.scatterplot(x='ram',y='px_height',hue='price_range',data = data1)
plt.show()

sns.set(color_codes=True)
plt.title('Price_range by Ram and Px_width', fontdict={'fontsize':15})
sns.scatterplot(x='ram',y='px_width',hue='price_range',data = data1)
plt.show()
#%%
#                                    Correlation

#%%
#%%
plt.figure(figsize=(14,9))
#sns.set_theme(style="white")
corr = data1.corr()
heatmap = sns.heatmap(corr, annot=True, cmap="Blues", fmt='.1g')
#%%
#%%
#Finding p value for battery_power

battery_power = df[['battery_power']]

print(battery_power.mean())
#%%

df1=pd.DataFrame(data1,columns=['battery_power'])
print(df1)
#%%
from scipy.stats import ttest_1samp
ttest,p_value=ttest_1samp(df1,1238)
print(p_value)
#%%
if p_value<.05:
    print("We are rejecting null hypothesis")
else:
    print("We are accepting null hypothesis")
#%%
#%%
#finding p value for int_memory

int_memory = df[['int_memory']]

print(int_memory.mean())
#%%

df2=pd.DataFrame(data1,columns=['int_memory'])
print(df2)
#%%
from scipy.stats import ttest_1samp
ttest,p_value=ttest_1samp(df2,32)
print(p_value)
#%%
if p_value<.05:
    print("We are rejecting null hypothesis")
else:
    print("We are accepting null hypothesis")
#%%
#finding p value for mobile_wt

mobile_wt = df[['mobile_wt']]

print(mobile_wt.mean())
#%%

df3=pd.DataFrame(data1,columns=['mobile_wt'])
print(df3)
#%%
from scipy.stats import ttest_1samp
ttest,p_value=ttest_1samp(df3,140)
print(p_value)
#%%
if p_value<.05:
    print("We are rejecting null hypothesis")
else:
    print("We are accepting null hypothesis")
#%%
##finding p value for pc

pc = df[['pc']]

print(pc.mean())
#%%

df4=pd.DataFrame(data1,columns=['pc'])
print(df4)
#%%
from scipy.stats import ttest_1samp
ttest,p_value=ttest_1samp(df4,10)
print(p_value)
#%%
if p_value<.05:
    print("We are rejecting null hypothesis")
else:
    print("We are accepting null hypothesis")
#%%
##finding p value for px_width

px_width = df[['px_width']]

print(px_width.mean())
#%%

df5=pd.DataFrame(data1,columns=['px_width'])
print(df5)
#%%
from scipy.stats import ttest_1samp
ttest,p_value=ttest_1samp(df5,1251)
print(p_value)
#%%
if p_value<.05:
    print("We are rejecting null hypothesis")
else:
    print("We are accepting null hypothesis")
#%%
#%%
##finding p value for ram

ram = df[['ram']]

print(ram.mean())
#%%

df6=pd.DataFrame(data1,columns=['ram'])
print(df6)
#%%
from scipy.stats import ttest_1samp
ttest,p_value=ttest_1samp(df6,2124)
print(p_value)
#%%
if p_value<.05:
    print("We are rejecting null hypothesis")
else:
    print("We are accepting null hypothesis")


#%%  ----------------CATEGORICAL ------------------------------------------
#%%
categorical = ['blue' , 'dual_sim' , 'four_g' , 'n_cores' , 'three_g' , 'touch_screen' ,'wifi' , 'price_range']
#%% Chi-square test
#Is 'three_g'  associated with 'four_g'?
             
import scipy.stats as stats
          
dataset=pd.crosstab(df['three_g' ],df['four_g'])
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
chi_square=sum([(o-e)**2/e for o,e in zip(Observed_value,expected_value)])
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
#%%
#%% Hypotheisis testing for 1 sample

#%%Hypothesis testing for 2 samples


import matplotlib
from matplotlib import pyplot
pyplot.figure(figsize=(14,6))
pyplot.hist(df['battery_power'])
pyplot.show()
#%%
import matplotlib
from matplotlib import pyplot
pyplot.figure(figsize=(14,6))
pyplot.hist(df['battery_power'])
pyplot.show()
#%%

df7=pd.DataFrame(data1,columns=['battery_power'])
print(df7)
#%%

df8=pd.DataFrame(data1,columns=['battery_power'])
print(df8)
#%%


print(df8.mean())
#%%One sample T test
import scipy
scipy.stats.ttest_1samp(df7,1200)#suppose the null hypothesis is 1237
#%%
if p_value<.05:
    print("We are rejecting null hypothesis")
else:
    print("We are accepting null hypothesis")
#%% 2 sample T test Indepependent
#null hypothesis is mean of sysBP and diaBP is equal
scipy.stats.ttest_ind(df7,df8)
#p value is very small.so,we can say that it rejects null hypothesis.


