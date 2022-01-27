import torch.nn as nn

class EmbeddingClassifierFactory:
    def __init__(self, embeddingSize):
        self.embeddingSize = embeddingSize

    def __call__(self, numClasses=10, *args, **kwargs):
        model = nn.Sequential(nn.Linear(self.embeddingSize, 50),
                              nn.LeakyReLU(),
                              nn.Linear(50, numClasses),
                              nn.Softmax(dim=1))
        return model


if __name__ == '__main__':
    import Data
    import numpy as np
    x_train, y_train, x_test, y_test = Data.load_mnist_mobilenet(numTest=10000, prefix='..')
    modelFactory = EmbeddingClassifierFactory(1280)
    errList = list()
    for i in range(3):
        model = modelFactory()
        model.fit(x_train, y_train, batch_size=64, epochs=30)
        metr = model.evaluate(x_test, y_test)
        f1 = np.mean(metr[2])
        errList.append(f1)
        print('test f1', f1)

    print('averaged', sum(errList) / len(errList))


# hidden: 10            f1: 0.941
# hidden: 0             f1: 0.942
# hidden: 100           f1: 0.943