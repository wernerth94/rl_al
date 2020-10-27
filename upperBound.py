import Classifier
from tensorflow import keras
import Data

x_train, y_train, x_test, y_test = Data.loadMNIST(numTest=5000)

es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=1)
avrg = 0
for run in range(1):
    print(run)
    model = Classifier.DenseClassifierMNIST()
    model.fit(x_train, y_train, batch_size=128, epochs=100, verbose=1, callbacks=es, validation_data=(x_test, y_test))
    print(model.evaluate(x_test, y_test))

#print('avrg', avrg / 5)