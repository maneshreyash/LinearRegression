#####################################################################################################################
#   CS 6375.003 - Assignment 1, Linear Regression using Gradient Descent
#   This is a simple starter code in Python 3.6 for linear regression using the notation shown in class.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   test - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class LinearRegression:
    def __init__(self, train):
        np.random.seed(1)
        # train refers to the training dataset
        # stepSize refers to the step size of gradient descent
        df = pd.read_csv(train)
        # #print(df)
        #
        # df[['Column1', 'Column2', 'Column3', 'Column4', 'Column5', 'Column6', 'Column7', 'Column8', 'Column9', 'Column10', 'Column11', 'Column12', 'Column13', 'Column14', 'Column15', 'Column16', 'Column17', 'Column18', 'Column19', 'Column20', 'Column21', 'Column22', 'Column23', 'Column24', 'Column25', 'Column26']] = df[['Column1', 'Column2', 'Column3', 'Column4', 'Column5', 'Column6', 'Column7', 'Column8', 'Column9', 'Column10', 'Column11', 'Column12', 'Column13', 'Column14', 'Column15', 'Column16', 'Column17', 'Column18', 'Column19', 'Column20', 'Column21', 'Column22', 'Column23', 'Column24', 'Column25', 'Column26']].replace('?', np.NaN)
        #
        #
        # cat = {"mazda" : 1, "mercedes-benz" : 2, "mercury" : 3, "mitsubishi" : 4, "alfa-romero" : 1, "audi" : 2, "bmw" : 3, }
        #
        # fuel_type = {"diesel" : 1, "gas" : 2}
        #
        #
        #
        # df = df.dropna()
        # print(df)

        df.insert(0, 'X0', 1)
        self.nrows, self.ncols = df.shape[0], df.shape[1]
        self.X = df.iloc[:, 0:(self.ncols-1)].values.reshape(self.nrows, self.ncols-1)
        self.y = df.iloc[:, (self.ncols-1)].values.reshape(self.nrows, 1)
        self.W = np.random.rand(self.ncols-1).reshape(self.ncols-1, 1)


    # TODO: Perform pre-processing for your dataset. It may include doing the following:
    #   - getting rid of null values
    #   - converting categorical to numerical values
    #   - scaling and standardizing attributes
    #   - anything else that you think could increase model performance
    # Below is the pre-process function


    # Below is the training function
    def train(self, epochs = 10, learning_rate = 0.05):
        # Perform Gradient Descent
        for i in range(epochs):
            # Make prediction with current weights

            h = np.dot(self.X, self.W)

            # Find error
            error = h - self.y

            # print(self.W)
            # print(self.X.T)
            self.W = self.W - (1 / self.nrows) * learning_rate * np.dot(self.X.T, error)

        return self.W, error

    # predict on test dataset
    def predict(self, test):
        testDF = pd.read_csv(test)
        testDF.insert(0, "X0", 1)
        nrows, ncols = testDF.shape[0], testDF.shape[1]
        testX = testDF.iloc[:, 0:(ncols - 1)].values.reshape(nrows, ncols - 1)
        testY = testDF.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        pred = np.dot(testX, self.W)
        error = pred - testY
        mse = (1/2.0*nrows) * np.dot(error.T, error)
        return mse


def preProcess(data):
    # dropping columns "symboling and normalized losses"
    df = pd.read_csv(data)

    df.drop(df.columns[[0, 1]], axis=1, inplace=True)

    df = df.where(df != '?', np.nan)
    df = df.dropna()

    lb_make = LabelEncoder()
    columns = [0, 1, 2, 3, 4, 5, 6, 12, 13, 15]
    for i in columns:
        df.iloc[:, i] = lb_make.fit_transform(df.iloc[:, i])
    for i in range(0, 24):
        df.iloc[:, i] = pd.to_numeric(df.iloc[:, i])
    print(df.iloc[:, 18])
    data = df.values
    return data


if __name__ == "__main__":

    dataset = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'

    data = pd.read_csv(dataset)
    data = preProcess(dataset)
    print(data.shape)
    np.random.shuffle(data)
    split = int(len(data) * .8)
    training, test = data[:split, :], data[split:, :]

    pd.DataFrame(training).to_csv("train1.csv")
    pd.DataFrame(test).to_csv("test1.csv")

    # preProcess("train1.csv")
    model = LinearRegression("train1.csv")

    W, e = model.train()
    mse = model.predict("test1.csv")
    print(mse)



