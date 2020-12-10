import tensorflow.keras as K
import tensorflow
import numpy as np
import Data, Classifier, Environment, Agent
from Evaluation import scoreAgent
from Misc import saveFile
import time, os
import gc

import config.batchConfig as c

preprocess_input = K.layers.experimental.preprocessing.Resizing(64, 64)
feature_extractor = K.applications.MobileNetV2(input_shape=(64, 64, 3),
                                               include_top=False,
                                               weights="imagenet")
feature_extractor.trainable = False
embeddingModel = K.models.Sequential([
    K.Input([28, 28, 3]),
    preprocess_input,
    feature_extractor,
    K.layers.GlobalAveragePooling2D()
])
embeddingModel.summary()

x_train, y_train, x_test, y_test = Data.loadMNIST(color=True, prefix='..')

def embed(x, embedding_model, batch_size=512):
    i = 0
    n = len(x)
    result = np.zeros([n, 1280])
    while i < n:
        print(i, '/', n)
        offset = min(i+batch_size, n)
        result[i:offset] = embedding_model(x[i:offset])
        i += batch_size
    gc.collect()
    return result


x_test = embed(x_test, embeddingModel)
x_train = embed(x_train, embeddingModel)

dataset = (x_train, y_train, x_test, y_test)
np.savez(os.path.join('../../datasets/mnist_mobileNetV2'), x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
