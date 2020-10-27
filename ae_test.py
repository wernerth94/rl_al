import tensorflow.keras as K
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt

state = K.layers.Input(6)
hState = K.layers.Dense(6, activation='tanh', kernel_regularizer=K.regularizers.L2(0.1))(state)
dropState = K.layers.Dropout(0.1)(hState)

action = K.layers.Input(2)
hAction = K.layers.Dense(2, activation='tanh', kernel_regularizer=K.regularizers.L2(0.1))(action)
dropAction = K.layers.Dropout(0.1)(hAction)

con = K.layers.concatenate([dropState, dropAction])
hidden = K.layers.Dense(3, activation='tanh', kernel_regularizer=K.regularizers.L2(0.1))(con)
drop = K.layers.Dropout(0.1)(hidden)
out = K.layers.Dense(6)(drop)

model = K.models.Model(inputs=[state, action], outputs=out)
model.compile(optimizer=K.optimizers.SGD(learning_rate=0.00001),
              loss=K.losses.hinge,
              metrics=[K.losses.mae])

s = np.load('memoryBacklog/iris_0-100k/state.npy')
sPrime = np.load('memoryBacklog/iris_0-100k/newState.npy')
a = np.load('memoryBacklog/iris_0-100k/actions.npy')
print('action mean', np.mean(a))
a = K.utils.to_categorical(a)

idx = np.arange(len(s))
np.random.shuffle(idx)
trainIds = idx[:80000]
testIds = idx[80000:]

s_train = s[trainIds]; s_test = s[testIds]
sPrime_train = sPrime[trainIds]; sPrime_test = sPrime[testIds]
a_train = a[trainIds]; a_test = a[testIds]

plat = K.callbacks.ReduceLROnPlateau(monitor='val_mean_absolute_error', patience=10, factor=0.5, verbose=1)
#es = K.callbacks.EarlyStopping(monitor='val_mean_absolute_error', mode='min', patience=15)
trainHist = model.fit([s_train, a_train], sPrime_train, epochs=200, batch_size=32, validation_data=([s_test, a_test], sPrime_test), callbacks=[plat])

for i in range(10):
    print(s_test[i], a_test[i], '->', model.predict([s_test[i].reshape(1, -1), a_test[i].reshape(1, -1)]))

plt.plot(trainHist.history['val_mean_absolute_error'])
plt.show()