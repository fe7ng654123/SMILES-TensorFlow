
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import BatchNormalization, Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D , AveragePooling2D
from tensorflow.python.keras import regularizers

epoch = 2

train_data = pickle.load(open("SR-ARE-train/names_onehots.pickle", "rb"))
train_names = train_data['names']
train_smiles = train_data['onehots'].reshape(-1, 70, 325, 1)
train_labels = np.loadtxt('SR-ARE-train/names_labels.txt', delimiter=',', usecols=1)

test_data = pickle.load(open("SR-ARE-test/names_onehots.pickle", "rb"))
test_names = np.array(test_data['names'])
test_smiles = np.array(test_data['onehots']).reshape(-1, 70, 325, 1)
test_labels = np.loadtxt('SR-ARE-test/names_labels.txt', delimiter=',', usecols=1)

model = Sequential()

model.add(Conv2D(33, (3, 3), input_shape=train_smiles.shape[1:]))
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


# class weight for dealing imbalanced data
class_weight = {0: 1.,
                1: 5.5}


model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.00003),
              metrics=["Recall"])



model.fit(train_smiles, train_labels, batch_size=32, epochs=epoch, validation_data=(test_smiles, test_labels),
          class_weight=class_weight)

print(model.evaluate(test_smiles, test_labels))
model.save_weights('model.h5')

