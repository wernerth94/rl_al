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