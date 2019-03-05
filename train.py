import numpy as np
import pandas as pd
import os
from pprint import pprint
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

train_data = pd.read_csv('emnist-balanced-train.csv', header=None)
test_data = pd.read_csv('emnist-balanced-test.csv', header=None)

train_X = train_data.iloc[:, 1:]
train_y = train_data.iloc[:, 0]
test_X = test_data.iloc[:, 1:]
test_y = test_data.iloc[:, 0]

img_size = 28
train_X = np.transpose(train_X.values.reshape(train_X.shape[0], img_size, img_size, 1), axes=[0,2,1,3])/255
test_X = np.transpose(test_X.values.reshape(test_X.shape[0], img_size, img_size, 1), axes=[0,2,1,3])/255

batch_size = 128
num_classes = 47
epochs = 10

train_y = keras.utils.to_categorical(train_y, num_classes)
test_y = keras.utils.to_categorical(test_y, num_classes)

model = Sequential()

model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(train_X, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(test_X, test_y))

score = model.evaluate(test_X, test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


#Save the model
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")