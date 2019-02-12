
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

dataset = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'

df = pd.read_csv(dataset)

df.drop(df.columns[[0, 1]], axis=1, inplace=True)

df = df.where(df != '?', np.nan)
df = df.dropna()

lb_make = LabelEncoder()
columns = [0, 1, 2, 3, 4, 5, 6, 12, 13, 15]
for i in columns:
    df.iloc[:, i] = lb_make.fit_transform(df.iloc[:, i])
for i in range(0, 24):
    df.iloc[:, i] = pd.to_numeric(df.iloc[:, i])
data_X = df.iloc[:, :-1]
data_y = df.iloc[:, -1]

# data_X = preprocessing.scale(data_X.values)
# data_y = preprocessing.scale(data_y.values)

# Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.33, random_state=42)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)
print(y_test)
print(y_pred)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('R squared value %.2f' % r2_score(y_test, y_pred))

# Plot outputs
plt.scatter(X_test[:, 20], y_test,  color='black')
plt.plot(X_test[:, 20], y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()