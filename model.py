import cv2
import json
import random
import math
import argparse
import time
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.layers import Convolution2D, Conv2D, ELU, Flatten, Dense, Dropout, Lambda, Activation, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def nvidia_model():
  row, col, depth = 70, 320, 3
  model = Sequential()

  # normalize image values between -.5 : .5
  model.add(Lambda(lambda x: x/255 - .5, input_shape=(row, col, depth), output_shape=(row, col, depth)))

  #valid border mode should get rid of a couple each way, whereas same keeps
  model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid'))
  # Use relu (non-linear activation function)
  model.add(Activation('relu'))
  model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid'))
  model.add(Activation('relu'))
  model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid'))
  model.add(Activation('relu'))
  model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
  model.add(Activation('relu'))
  model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))

  model.add(Flatten())

  model.add(Dropout(.5))
  model.add(Activation('relu'))

  model.add(Dense(100))
  # model.add(Dropout(.3))
  model.add(Activation('relu'))

  model.add(Dense(50))
  model.add(Activation('relu'))

  model.add(Dense(10))
  model.add(Activation('relu'))

  model.add(Dense(1))

  #compile with normal adam optimizer (loss .001) and return
  model.compile(loss='mse', optimizer='adam')
  model.summary()
  return model

def my_generator(X, y, batch_size, num_per_epoch):
  while True:
    #X, y = shuffle(X, y)
    smaller = min(len(X), num_per_epoch)
    iterations = math.ceil(smaller/batch_size)
    for i in range(iterations):
      start, end = i * batch_size, (i + 1) * batch_size
      yield (X[start:end], y[start:end])


batch = 128
epoch = 5
epochsize = 80000
destfile = "./model"

start_time = time.time()
print("loading data...")

X_train = np.load("./X_train.npy")
y_train = np.load("./y_train.npy")

X_val = np.load("./X_val.npy")
y_val = np.load("./y_val.npy")

print("data loaded: ", time.time() - start_time, "s")
print('X_train and y_train', X_train.shape, y_train.shape, type(X_train))
print('X_val and y_val', X_val.shape, y_val.shape)

model = nvidia_model()


    #with open('models/nvidia_3_15.json', 'r') as jfile:
    #  model = model_from_json(json.load(jfile))

adam = Adam(lr=.0001)
model.compile(optimizer=adam, loss="mse")
    #weights_file = 'models/nvidia_3_15.h5'
    #model.load_weights(weights_file)

for i in range(1, 1 + epoch):
    print('epoch ', i)
    score = model.fit_generator(
    generator=my_generator(X=X_train, y=y_train, batch_size=batch, num_per_epoch=epochsize),
    nb_epoch=1,
    samples_per_epoch=min(epochsize, len(X_train)),
    validation_data=my_generator(X=X_val, y=y_val, batch_size=batch, num_per_epoch=epochsize),
    nb_val_samples=y_val.shape[0])

    epoch = str(i)
    model.save_weights(destfile + '_' + epoch +'.h5', True)
    with open(destfile + '_' + epoch + '.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)
    print('saved model as', destfile + '_' + epoch)
print("total time used: ", (time.time() - start_time)/60)
