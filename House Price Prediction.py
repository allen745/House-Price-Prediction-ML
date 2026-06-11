# importing the dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn  import metrics

# importing the california house data set

house_price_dataset = pd.read_csv(r'C:\Users\allen\OneDrive\Desktop\understanding\data\housing.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
# print(house_price_dataset)
print(house_price_dataset.head())
print(house_price_dataset.tail())
#print columns
print(house_price_dataset.columns)


# checking nimber of rows and column in datafram
print(house_price_dataset.shape)
print(house_price_dataset.describe())

# check for Null values
print(house_price_dataset.isnull().sum())

# fill missing values with mean
house_price_dataset['total_bedrooms'] = house_price_dataset['total_bedrooms'].fillna(house_price_dataset['total_bedrooms'].mean())

# verify
print(house_price_dataset.isnull().sum())
print(house_price_dataset.describe())

# understanding the correlation between various features of dataset.
# positive correlation
# negative correlation

correlation =house_price_dataset.corr(numeric_only = True)
# constructing heat map to understand correlation
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar= True, square= True, annot= True, fmt= '.1f', annot_kws= {'size':10}, cmap='Blues')
plt.show()

# splitting data and prices
x = house_price_dataset.drop(columns=['median_house_value', 'ocean_proximity'])
y = house_price_dataset['median_house_value']
# print(x)
# print(y)

# splitting the data into training data and test data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 2)
print(X_train.shape,X_test.shape)

# MODA TRAINING
# XGBoost Regression
# loading the modal
modal = XGBRegressor()

# training the modal with x train
modal.fit(X_train, y_train)

# evalution
# prediction on training data
# Accuracy for prediction on training data
training_data_predction = modal.predict(X_train)
print(training_data_predction)

# R square error
score1= metrics.r2_score(y_train, training_data_predction)

# mean absolute error
score2= metrics.mean_absolute_error(y_train, training_data_predction)

print("R squared error Train: ", score1)
print("Mean absolute error Train: ", score2)

# visualizing the actual price and predicted price
plt.scatter(y_train, training_data_predction)
plt.xlabel('actual price')
plt.ylabel('predicted price')
plt.title('actual price vs predicted price')
plt.show()
# predction on test data
testing_data_prediction = modal.predict(X_test)

# R square error
score1= metrics.r2_score(y_test, testing_data_prediction)

# mean absolute error
score2= metrics.mean_absolute_error(y_test, testing_data_prediction)

print("R squared error Test: ", score1)
print("Mean absolute error Test: ", score2)



