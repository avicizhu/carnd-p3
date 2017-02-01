import cv2
import json
import random
import math
import argparse
import numpy as np
from scipy import misc
import time
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
#from process_data import crop_images, resize_images, show_images, change_brightness, flip_half, flip_X, flip_y, translate

np_dir = './'
model_dir = './'

def get_model(time_len=1):
  ch, row, col = 3, 160, 320  # camera format
  shape = (row, col, ch)

  model = Sequential()
  model.add(Lambda(lambda x: x/127.5 -1., input_shape=shape, output_shape=shape))
  model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(Flatten())
  model.add(Dropout(.2))
  model.add(ELU())
  model.add(Dense(512))
  model.add(Dropout(.5))
  model.add(ELU())
  model.add(Dense(1))

  model.compile(optimizer="adam", loss="mse")

  return model

'''
add in validation generator
'''
def val_generator(X, y, batch_size, num_per_epoch):
  while True:
    # X, y = shuffle(X, y)
    smaller = min(len(X), num_per_epoch)
    iterations = math.ceil(smaller/batch_size)
    for i in range(iterations):
      start, end = i * batch_size, (i + 1) * batch_size
      yield (X[start:end], y[start:end])

'''
create generator to create augmented images
'''
def my_generator(X, y, batch_size, num_per_epoch):
  while True:
    #X, y = shuffle(X, y)
    #print('range is', ceil(num_per_epoch/batch_size))
    smaller = min(len(X), num_per_epoch)
    iterations = math.ceil(smaller/batch_size)
    for i in range(iterations):
      start, end = i * batch_size, (i + 1) * batch_size
      yield (X[start:end], y[start:end])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model to train steering angles')
    parser.add_argument('--batch', type=int, default=128, help='Batch size.')
    parser.add_argument('--epoch', type=int, default=5, help='Number of epochs.')
    parser.add_argument('--epochsize', type=int, default=43394, help='How many images per epoch.')
    parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='?multiple path out.')
    parser.add_argument('--features', type=str, default=np_dir + 'X_train.npy', help='File where features .npy found.')
    parser.add_argument('--labels', type=str, default=np_dir + 'y_train.npy', help='File where labels .npy found.')
    parser.add_argument('--destfile', type=str, default=model_dir + 'comma', help='File where model found')

    parser.set_defaults(skipvalidate=False)
    parser.set_defaults(loadweights=False)
    args = parser.parse_args()

    start_time = time.time()
    print("loading data...")

    X_train = np.load("./X_train.npy")
    y_train = np.load("./y_train.npy")

    X_val = np.load("./X_val.npy")
    y_val = np.load("y_val.npy")

    print("data loaded: ", time.time() - start_time, "s")
    print('X_train and y_train', X_train.shape, y_train.shape)
    print('X_val shape', X_val.shape)

    model = get_model()

    #with open('models/nvidia_3_15.json', 'r') as jfile:
    #  model = model_from_json(json.load(jfile))

    adam = Adam(lr=.0001)
    model.compile(optimizer=adam, loss="mse")
    #weights_file = 'models/nvidia_3_15.h5'
    #model.load_weights(weights_file)

    for i in range(1, 1 + args.epoch):
        print('epoch ', i)
        score = model.fit_generator(
        generator=my_generator(X=X_train, y=y_train, batch_size=args.batch, num_per_epoch=args.epochsize),
        nb_epoch=1,
        samples_per_epoch=min(args.epochsize, len(X_train)),
        validation_data=val_generator(X=X_val, y=y_val, batch_size=args.batch, num_per_epoch=args.epochsize),
        nb_val_samples=800)

        epoch = str(i + 1)
        model.save_weights(args.destfile + '_' + epoch +'.h5', True)
        with open(args.destfile + '_' + epoch + '.json', 'w') as outfile:
            json.dump(model.to_json(), outfile)
        print('saved model as', args.destfile + '_' + epoch)
    print("total time used: ", (time.time() - start_time)/60)
