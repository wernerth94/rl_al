from sklearn.datasets import load_svmlight_file

def load_dataset(name)->tuple:
    # this returns a tuple of (x_train, y_train, x_test, y_test)
    train = load_svmlight_file("Datasets/mnist_train.bz2")
    test = load_svmlight_file("Datasets/mnist_test.bz2")

    x_train = train[0]
    y_train = train[1]
    x_test = test[0]
    y_test = test[1]

    x_train = x_train.toarray()
    x_test = x_test.toarray()

    pass