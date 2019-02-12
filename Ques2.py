#####################################################################################################################
#   CS 6375.003 - Assignment 1, Linear Regression using Gradient Descent
#   @author : Shreyash Mane     ssm170730
#           : Supraja Ponnur    sxp179130
#   Designed the preProcess function to modify the dataset to fit the necessary constraints of Linear Regression.
#
#####################################################################################################################

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# DataSet URL
dataset = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
# reading the data into a csv file
df = pd.read_csv(dataset)
# dropping the first 2 columns which are actually the predicted ones in the UCI repository.

df.drop(df.columns[[0, 1]], axis=1, inplace=True)
# Detecting the missing values which are marked as '?' and replacing with 'nan'
df = df.where(df != '?', np.nan)
# Dropping the 'nan' i.e. missing values
df = df.dropna()
# using Label Encoder to convert categorical data into numerical data fit for the Regression model
lb_make = LabelEncoder()
# Selecting the variables which have categorical values
columns = [0, 1, 2, 3, 4, 5, 6, 12, 13, 15]
for i in columns:
    df.iloc[:, i] = lb_make.fit_transform(df.iloc[:, i])
# All the variables are of d-type initially; converting to numeric values
for i in range(0, 24):
    df.iloc[:, i] = pd.to_numeric(df.iloc[:, i])
data_X = df.iloc[:, :-1]
data_y = df.iloc[:, -1]

# Splitting the data in the ratio of  80:20 Train:Test respectively
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.33, random_state=42)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('R squared value %.2f' % r2_score(y_test, y_pred))

# Plot outputs
plt.scatter(X_test.iloc[:, 8], y_test,  color='blue')
plt.scatter(X_test.iloc[:, 8], y_pred,  color='red')

plt.plot(X_test.iloc[:, 8], y_pred, color='green', linewidth=1)

plt.show()
