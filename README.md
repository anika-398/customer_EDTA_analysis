# customer_EDTA_analysis
Data cleaning, EDA, hypothesis testing, and statistical analysis, all centered around a customer dataset.
#Description: 
The main purpose of this project is to implement Python modules, data frames, and functions. We were given some datasets on great customers, heart disease and mobile price train for our project. There are some main parts in the project: finding p values, association testing and analyzing, data exploration etc. For these works we had to do data cleaning, data processing, exploration, Hypothesis testing, Chi-square testing etc.
# Data Preprocessing 
Handling Missing values: Heart Disease 0 , customer 7614, mobile price 645.
#Dataset Characteristics and Exploratory Data Analysis (EDA)
Checking the shape of the dataset, rows, columns,  the unique values in the dataset. And from this slected unique values from each of the columns. Determinig categorical calues.  The statistical description of the dataframe.  
taking a sample from this dataset using sample() function for further calculations.
 Here, we can see a clear difference between the count, mean, standard deviation, minimum ,maximum values, 25th percentile, 50 th percentile and 75 th percentile population dataset and the sample dataset. We use the sample dataset to do the EDA .
 
 #Plots generation using matplotlib, seaborn :

 # P-value, Test of association
 Determining P-value for heart_disease, customer and mobile train: 
P value is a number describing how likely it is that data would have occurred under the null hypothesis of a statistical  test. For determining P-value, First, We select some features whose values are normalized or almost normalized from the dataset. 

#Chi-square test 
#Is male associated with currentSmoker? 

#For mobile_price_train, Categorical variables are 'blue' , 'dual_sim' , 'four_g' , 'n_cores' , 'three_g' , 'touch_screen' ,'wifi' , 'price_range'. 
Is 'three_g' associated with 'four_g'? 

# Determining Chi-Square test for great_customer: 
#Is sex associated with work class?    

#Hypothesis Testing

Question:Is there a significant relationship between #sysBP and diaBp in a given dataset? 
Question: Is there a significant difference in the average battery power and Ram in a given dataset? 
