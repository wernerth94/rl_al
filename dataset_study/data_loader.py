import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MinMaxScaler

def load_dataset(name)->tuple:
    # this returns a tuple of (x_train, y_train, x_test, y_test)
    if name == "mnist":
        train = load_svmlight_file("dataset_study/Datasets/mnist_train.bz2", n_features=784)
        test = load_svmlight_file("dataset_study/Datasets/mnist_test.bz2", n_features=784)

    elif name == "protein":
        train = load_svmlight_file("dataset_study/Datasets/protein_train.bz2", n_features=357)
        test = load_svmlight_file("dataset_study/Datasets/protein_test.bz2", n_features=357)

    elif name == "dna":
        train = load_svmlight_file("dataset_study/Datasets/dna_train.txt", n_features=180)
        test = load_svmlight_file("dataset_study/Datasets/dna_test.txt", n_features=180)

    elif name == "a8a":
        train = load_svmlight_file("dataset_study/Datasets/a8a_train.txt", n_features=123)
        test = load_svmlight_file("dataset_study/Datasets/a8a_test.txt", n_features=123)

    elif name == "splice":
        train = load_svmlight_file("dataset_study/Datasets/splice_train.txt", n_features=60)
        test = load_svmlight_file("dataset_study/Datasets/splice_test.txt", n_features=60)

    elif name == "usps":
        train = load_svmlight_file("dataset_study/Datasets/usps_train.bz2", n_features=256)
        test = load_svmlight_file("dataset_study/Datasets/usps_test.bz2", n_features=256)

    x_train = train[0]
    y_train = train[1].astype(int)
    x_test = test[0]
    y_test = test[1].astype(int)

    x_train = x_train.toarray()
    x_test = x_test.toarray()

    scaler = MinMaxScaler()
    scaler.fit(x_train)
    scaler.fit(x_test)

    mask = y_train == -1
    y_train[mask] += 1

    mask = y_test == -1
    y_test[mask] += 1

    one_hot_train = np.zeros((len(y_train), y_train.max()+1))
    one_hot_train[np.arange(len(y_train)), y_train] = 1

    one_hot_test = np.zeros((len(y_test), y_test.max()+1))
    one_hot_test[np.arange(len(y_test)), y_test] = 1

    return (x_train, one_hot_train, x_test, one_hot_test)