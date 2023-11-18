# Churn-Analysis-using-Logistic-Regression

# Table of contents

1. Import Data
2. Inspecting and creating heatmap of missing values
3. Data Cleaning and creating heatmap of a clean data
4. Encoding categorical variables
5. Data visualizations
6. Perform a diagnostic analysis
7. Run the ML Model
8. Saving & Running the Model
9. Dealing with Class Imbalance
10. Predictions

    
Churn Prediction Problem Statement We will be working on a churn dataset. Churned Customers are those who have decided to end their relationship with their existing company.

The streaming app is a service-providing company that provides customers with a one-year subscription plan for their product. The company wants to know if the customers will renew the subscription for the coming year or not.

Business Objective
Predicting a qualitative response for observation can be referred to as classifying that observation since it involves assigning the observation to a category or class. Classification forms the basis for Logistic Regression. Logistic Regression is a supervised algorithm used to predict a dependent variable that is categorical or discrete. Logistic regression models the data using the sigmoid function.

Churned Customers are those who have decided to end their relationship with their existing company. In our case study, we will be working on a churn dataset.

Streaming App is a service-providing company that provides customers with a one-year subscription plan for their product. The company wants to know if the customers will renew the subscription for the coming year or not.


Data Description

The CSV consists of around 2000 rows and 16 columns
Features:
1.	Year
2.	Customer_id-uniqueid
3.	Phone_no-customerphoneno
4.	Gender-Male/Female
5.	Age
6.	No of days subscribed-thenumberof days sincethe subscription
7.	Multi-screen-doesthecustomerhaveasingle/multiplescreensubscription
8.	Mail subscription - customerreceivemailsor not
9.	Weeklyminswatched-number of minuteswatched weekly
10.	Minimumdailymins-minimumminutes watched
11.	Maximumdailymins- maximum minutes watched
12.	Weeklynights maxmins-number of minuteswatchedat nighttime
13.	Videoswatched-totalnumber ofvideos watched
14.	Maximum_days_inactive-dayssinceinactive
15.	Customer supportcalls-number of customer supportcalls
16.	Churn-
●	1-Yes
●	0-No


Techstack
	Language-Python
	Libraries-numpy,pandas,matplotlib,seaborn,sklearn,pickle,imblearn,statsmodel



Aim

Build a Logistic Regression model on the given dataset to determine whether the customer will churn or not.
