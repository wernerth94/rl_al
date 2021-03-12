import tensorflow.keras as K
import tensorflow_addons as tfa
import numpy as np
import Data
import os, gc

# preprocess_input = K.layers.experimental.preprocessing.Resizing(64, 64)
# feature_extractor = K.applications.MobileNetV2(input_shape=(64, 64, 3),
#                                                include_top=False,
#                                                weights="imagenet")
# feature_extractor.trainable = False
# embeddingModel = K.models.Sequential([
#     K.Input([32, 32, 3]),
#     preprocess_input,
#     feature_extractor,
#     K.layers.GlobalAveragePooling2D()
# ])
# embeddingModel.summary()

embeddingModel = K.models.Sequential([
        K.layers.Input([28, 28, 1]),
        K.layers.Conv2D(32, 3, strides=(2,2), activation='relu', kernel_regularizer=K.regularizers.l2(0.01)),
        K.layers.Dropout(0.5),
        K.layers.Conv2D(12, 3, strides=(2,2), activation='relu', kernel_regularizer=K.regularizers.l2(0.01)),
        K.layers.Dropout(0.5),
        K.layers.Conv2D(12, 3, activation='relu', kernel_regularizer=K.regularizers.l2(0.01)),
        K.layers.Flatten()
    ])
embeddingModel.summary()
head = K.models.Sequential([
        K.layers.Dense(24, activation=K.activations.relu, kernel_regularizer=K.regularizers.l2(0.01)),
        K.layers.Dense(10, activation='softmax')
    ])
model = K.models.Sequential([
    embeddingModel,
    head
])
opt = tfa.optimizers.RectifiedAdam()
model.compile(optimizer=tfa.optimizers.Lookahead(opt),
              loss=K.losses.categorical_crossentropy)

x_train, y_train, x_test, y_test = Data.loadMNIST(numTest=-1, prefix='..')

es = K.callbacks.EarlyStopping()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=64, callbacks=[es])

def embed(x, embedding_model, batch_size=512):
    i = 0
    n = len(x)
    result = np.zeros([ n, embedding_model.output.shape[1] ])
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
np.savez(os.path.join('../../datasets/mnist_embedSmall'), x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
