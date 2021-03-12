import tensorflow
import tensorflow.keras as keras
import tensorflow_addons as tfa

def metrics(numClasses):
    return [keras.metrics.categorical_accuracy,
            tfa.metrics.F1Score(numClasses)]
            #keras.metrics.Precision(),
            #keras.metrics.Recall()]


def ImageClassifier(inputShape=[28,28,1], numClasses=10, l2_reg=0.02):
    model = keras.models.Sequential([
        keras.layers.Input(inputShape),
        #keras.layers.Conv2D(64, 3, strides=[3, 3], activation='relu', kernel_regularizer=keras.regularizers.l2(l2_reg)),
        keras.layers.Conv2D(32, 3, activation='relu', kernel_regularizer=keras.regularizers.l2(l2_reg)),
        keras.layers.Flatten(),
        keras.layers.Dense(24, activation=keras.activations.relu, kernel_regularizer=keras.regularizers.l2(l2_reg)),
        keras.layers.Dense(numClasses, activation='softmax')
    ])
    opt = tfa.optimizers.RectifiedAdam()
    model.compile(optimizer=tfa.optimizers.Lookahead(opt),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=metrics(numClasses))
    return model



def ImageClassifierCifar(inputShape=[32,32,3], numClasses=10):
    model = keras.models.Sequential([
        keras.layers.Input(inputShape),
        keras.layers.Conv2D(64, 3, strides=[3, 3], activation='relu'),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.Conv2D(16, 3, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(24, activation=keras.activations.relu),
        keras.layers.Dense(numClasses, activation='softmax')
    ])
    model.compile(optimizer=tensorflow.keras.optimizers.Adam(),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=metrics(numClasses))
    return model


def SimpleClassifier(inputShape, numClasses):
    model = keras.models.Sequential([
        keras.layers.Input(inputShape),
        keras.layers.Dense(12, activation=keras.activations.relu),
        keras.layers.Dense(numClasses, activation='softmax')
    ])
    opt = tfa.optimizers.RectifiedAdam()
    model.compile(optimizer=opt,
                  loss=keras.losses.categorical_crossentropy,
                  metrics=metrics(numClasses))
    return model


def DenseClassifierMNIST(inputShape=[28,28,1], numClasses=10):
    model = keras.models.Sequential([
        keras.layers.Input(inputShape),
        keras.layers.Flatten(),
        keras.layers.Dense(48, activation=keras.activations.relu),
        keras.layers.Dense(24, activation=keras.activations.relu),
        keras.layers.Dense(numClasses, activation='softmax')
    ])
    opt = tfa.optimizers.RectifiedAdam()
    model.compile(optimizer=tfa.optimizers.Lookahead(opt),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=metrics(numClasses))
    return model



def trafficClassifier(inputShape=[256], numClasses=43):
    model = keras.models.Sequential([
        keras.Input(shape=inputShape),
        keras.layers.Dense(numClasses, activation='softmax')
    ])
    opt = tfa.optimizers.RectifiedAdam()
    model.compile(optimizer=tfa.optimizers.Lookahead(opt),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=metrics(43))
    return model


class EmbeddingClassifier:
    def __init__(self, embeddingSize):
        self.embeddingSize = embeddingSize

    def __call__(self, numClasses=10, *args, **kwargs):
        model = keras.models.Sequential([
            keras.Input(shape=[self.embeddingSize]),
            #keras.layers.Dense(100, activation='relu'),
            keras.layers.Dense(numClasses, activation='softmax')
        ])
        opt = tfa.optimizers.RectifiedAdam()
        model.compile(optimizer=tfa.optimizers.Lookahead(opt),
                      loss=keras.losses.categorical_crossentropy,
                      metrics=metrics(numClasses))
        return model


if __name__ == '__main__':
    import Data
    import numpy as np
    x_train, y_train, x_test, y_test = Data.load_mnist_mobilenet(numTest=10000, prefix='..')
    modelFactory = EmbeddingClassifier(1280)
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