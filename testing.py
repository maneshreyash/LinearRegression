import numpy
import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
    data = pd.read_csv(dataset_url)
    data = data.iloc[:].values
    numpy.random.shuffle(data)
    print(len(data))
    split = int(len(data) * .8)
    training, test = data[:split, :], data[split:, :]
    print[len(training)]
    print(training.shape)