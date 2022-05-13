import numpy as np
import tensorflow as tf
import pickle
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import BatchNormalization, Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D


test_data = pickle.load(open("../SR-ARE-score/names_onehots.pickle", "rb"))
test_names = np.array(test_data['names'])
test_smiles = np.array(test_data['onehots']).reshape(-1, 70, 325, 1)
# test_labels = np.loadtxt('SR-ARE-test/names_labels.txt', delimiter=',', usecols=1)


model = Sequential()

model.add(Conv2D(33, (3, 3), input_shape=test_smiles.shape[1:]))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(10, (3, 3)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Dropout(0.5))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(150))
model.add(Activation('relu'))

# model.add(Dense(32))

model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              metrics=["Recall"])

model.load_weights('model.h5')
# print(model.evaluate(test_smiles, test_labels))
output = model.predict(test_smiles)
np.savetxt("labels.txt", np.where(output > 0.5, 1, 0), fmt='%i')
