import pickle
import numpy as np
import math
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import  Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, Conv2D,Lambda, ELU
from keras.regularizers import l2
from keras.layers.convolutional import Convolution2D

from keras.utils import np_utils
from keras.optimizers import Adam
from keras import backend as K

import json

import utils

from keras.callbacks import ModelCheckpoint


np.random.seed(1337)

#Given a list of files(from different datasets), randomly select files for training, testing, validation purpose
def split_train_test_validate_file_names():

    #Read a all the images file names and corresponding wheel value into a dictionary. Keep only useful info.
    file_wheel = utils.read_labels()
    files = list(file_wheel.keys())
    print(type(files))
    print(len(files))


    #Split 2000+ files for test and validation, rest of them use for training
    files_train, files_vali_test = train_test_split(files, test_size=0.07, random_state=42)
    #from the 2000+ selected files, split half for test, half for validation
    files_vali, files_test = train_test_split(files_vali_test, test_size=0.5, random_state=42)

    print("train: ", len(files_train))
    print("vali: ", len(files_vali))
    print("test: ", len(files_test))

    return files_train, files_vali, files_test


#Generate data for test and validation.
#Since they are fixed number of data. (Don't need yield)
def generate_train_test(files, file_wheel):

    #img_dir = data_dir + "IMG/"
    #img_prefix_center = "center"
    #img_prefix_left = "left"
    #img_prefix_right = "right"

    new_images = []
    new_wheels = []

    for file in files:
        file_path = file

        #Make sure the image exists to avoid exception
        if os.path.exists(file_path)  and (file in file_wheel):

            #Read images and preprocess it. See functions in the 'utils.py'
            img = utils.read_image(file_path)
            wheel = file_wheel[file]

            #Build the list for X and Y
            new_images.append(img)
            new_wheels.append(wheel)

    new_images = np.array(new_images)
    new_wheels = np.array(new_wheels)

    print(len(new_images))
    print(len(new_wheels))
    print("type(new_images)", type(new_images))
    print("type(new_wheels)", type(new_wheels))
    print("type(new_images[0])", type(new_images[0]))
    print("type(new_wheels)[0]", type(new_wheels[0]))

    return new_images, new_wheels

#Dynamicly generate train data. With yield.
#It saves a lot of memory when loading the images.
def generate_train(files, file_wheel):

    #yield 128 per batch
    batch_size = 128
    total_cnt = len(files)
    batch_num = int(total_cnt / batch_size)

    #Loop and serve as an unlimited data source
    while 1:
        #Shuffle before yield
        np.random.shuffle(files)
        for b1 in range(0, batch_num):

            print("Batch ", b1, " of ", batch_num)

            new_images = []
            new_wheels = []

            for b2 in range(0, batch_size):

                idx = b1 * batch_size + b2

                file = files[idx]
                file_path = file

                if os.path.exists(file_path)  and (file in file_wheel):
                    img = utils.read_image(file_path)
                    wheel = file_wheel[file]

                    new_images.append(img)
                    new_wheels.append(wheel)

            new_images = np.array(new_images)
            new_wheels = np.array(new_wheels)

            yield (new_images, new_wheels)


def train_model():

    #Get train, validation, test file list
    files_train, files_vali, files_test = split_train_test_validate_file_names()

    #Get file-wheel value dictionary
    file_wheel = utils.read_labels()

    #Generate validation data
    X_validation, Y_validation = generate_train_test(files_vali,file_wheel)
    #Generate test data
    X_test, Y_test = generate_train_test(files_test, file_wheel)


    model = Sequential()

    #original
    '''
    model.add(Conv2D(24, 5,5, input_shape=(66,200,3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add((Dropout(0.5)))
    model.add(Activation('relu'))
    model.add(Conv2D(36, 5,5, activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add((Dropout(0.5)))
    model.add(Activation('relu'))
    #model.add(Conv2D(48, 5,5, activation='relu'))
    model.add(Conv2D(48, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add((Dropout(0.5)))
    model.add(Activation('relu'))

    model.add(Conv2D(64, 3,3, activation='relu'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3,3, activation='relu'))
    model.add(Activation('relu'))


    model.add(Flatten())
    #model.add(Dense(1164, activation='relu'))
    model.add(Dense(110,activation='softmax'))
    model.add(Dense(55,activation='softmax'))
    model.add(Dense(11,activation='softmax'))
    model.add(Dense(1,name='output', activation='tanh'))

    model.summary()

    model.compile(loss='mse', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

    '''

    #-----------------------

    #The architecture of the model
    INIT = 'glorot_uniform'
    keep_prob = 0.2
    reg_val = 0.01
    ch, row, col = 3, 66, 200  # camera format

    model.add(Lambda(lambda x: x / 1.0 - 0.,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))

    #(3,66,200) => (24,5,5)
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", init=INIT, W_regularizer=l2(reg_val)))
    # W_regularizer=l2(reg_val)
    model.add(ELU())
    model.add(Dropout(keep_prob))

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", init=INIT))
    model.add(ELU())
    model.add(Dropout(keep_prob))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", init=INIT))
    model.add(ELU())
    model.add(Dropout(keep_prob))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init=INIT))
    model.add(ELU())
    model.add(Dropout(keep_prob))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init=INIT))
    model.add(ELU())
    model.add(Dropout(keep_prob))

    model.add(Flatten())

    model.add(Dense(100))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Dense(50))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(ELU())

    model.add(Dense(1))

    #Compile model
    model.compile(optimizer="adam", loss="mse")  # , metrics=['accuracy']

    checkpoint = ModelCheckpoint('checkpoints/best_save' + '-{epoch:02d}-{val_loss:.4f}',
                                 monitor='val_loss', verbose=0, save_best_only=True,
                                 save_weights_only=False, mode='auto')


    #----------------------

    #Run the training
    history = model.fit_generator(generate_train(files_train,file_wheel),128*200,5,verbose=1,validation_data=(X_validation,Y_validation),nb_val_samples=1000, callbacks=[checkpoint])

    #Save the model
    model_j = model.to_json()
    with open("./model.json","w") as jf:
        json.dump(model_j,jf)
    #model.save("./model_save.model",True)

    #Save the model weights
    model.save_weights("./model.h5",True)

    #Evaluate the model with test data.
    history2=model.evaluate(X_test, Y_test)
    print(history2)

train_model()










