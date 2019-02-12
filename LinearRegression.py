#####################################################################################################################
#   CS 6375.003 - Assignment 1, Linear Regression using Gradient Descent
#   @author : Shreyash Mane     ssm170730
#           : Supraja Ponnur    sxp179130
#   Designed the preProcess function to modify the dataset to fit the necessary constraints of Linear Regression.
#
#####################################################################################################################


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Class Linear Regression to fit a dataset into a linear regression model and
# predict the values from train data provided with test data
class LinearRegression:
    def __init__(self, train):
        np.random.seed(1)
        # train refers to the training dataset
        # stepSize refers to the step size of gradient descent
        df = pd.read_csv(train)

        df.insert(0, 'X0', 1)
        self.nrows, self.ncols = df.shape[0], df.shape[1]
        self.X = df.iloc[:, 0:(self.ncols-1)].values.reshape(self.nrows, self.ncols-1)
        self.y = df.iloc[:, (self.ncols-1)].values.reshape(self.nrows, 1)
        self.W = np.random.rand(self.ncols-1).reshape(self.ncols-1, 1)

    # Below is the train function that trains on the dataset
    # @param: self: The dataset from initialised function
    #        epochs: Number of Iterations
    #        learning_rate : The value of alpha i.e. the rate at which we take steps
    # @return: The newly computed value for W based on the hypothesis and the training error
    def train(self, epochs = 500, learning_rate = 0.25):
        # Perform Gradient Descent
        for i in range(epochs):
            # Make prediction with current weights
            h = np.dot(self.X, self.W)
            # Find error
            error = h - self.y
            self.W = self.W - (1 / self.nrows) * learning_rate * np.dot(self.X.T, error)
        return self.W, error

    # Below is the predict function
    # @param: self: The dataset from initialised function
    #         test: The testing dataset
    # @return: Mean Squared Error value
    def predict(self, test):
        testDF = pd.read_csv(test)
        testDF.insert(0, "X0", 1)
        nrows, ncols = testDF.shape[0], testDF.shape[1]
        testX = testDF.iloc[:, 0:(ncols - 1)].values.reshape(nrows, ncols - 1)
        testY = testDF.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        pred = np.dot(testX, self.W)

        error = pred - testY
        mse = (1/(2.0*nrows)) * np.dot(error.T, error)
        return mse


# A global function to preprocess the input data
# @param: in_data: Input data from the source URL
# @return: dat: The preprocessed data in the form of a numpy multidimensional array
def preprocess(in_data):

    df = pd.read_csv(in_data)
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
    # Converting the panda DataFrame to a numpy MultiDimensional Array
    dat = df.values
    return dat


# Main function to take in the data and call the functions
if __name__ == "__main__":
    # DataSet URL
    dataset = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
    # reading the data into a csv file
    data = pd.read_csv(dataset)
    # Function call to preprocess to clean and make the data compatible
    data = preprocess(dataset)
    # Shuffling the data so as to make it random for splitting to consider every possible variations of the data
    np.random.shuffle(data)
    # Splitting the data in the ratio of  80:20 Train:Test respectively
    split = int(len(data) * .8)
    training, test = data[:split, :], data[split:, :]
    # putting the data into csv files
    pd.DataFrame(training).to_csv("train1.csv")
    pd.DataFrame(test).to_csv("test1.csv")
    # Intialize the model for regression
    model = LinearRegression("train1.csv")
    # Training the data
    W, e = model.train()
    # Calculating MSE from predict
    mse = model.predict("test1.csv")
    print(mse)



