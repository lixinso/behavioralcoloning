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

np.random.seed(1337)

'''
#load data from multiple source
data_dirs = ["../session_data/","../data/","../datame2/"]

def read_labels():
    file_wheel_map = {}

    new_label_file = "./new_driving_log.csv"
    new_label_text = ""

    for data_dir in data_dirs:

        label_dir = data_dir + "driving_log.csv"

        cnt = 0
        with open(label_dir) as lf:
            for line in lf:

                cnt += 1

                if cnt  == 1:
                    continue

                columns = line.strip().split(",")
                if len(columns) > 4:
                    center_file = data_dir + columns[0].strip()#[4:]
                    #left_file = columns[1].strip()#[4:]
                    #right_file = columns[2].strip()#[4:]
                    #print(columns[3])
                    steering = float(columns[3].strip()) #int( round( float(columns[3].strip()) * 5.0 ) )

                    print(center_file, steering)
                    new_label_text += center_file + "," + str(steering) + "\n"

                    file_wheel_map[center_file] = steering

    with open(new_label_file,"w") as nf:
        nf.write(new_label_text)

    return file_wheel_map
'''

def read_imgs(file_wheel_map):


    img_dir = data_dir + "IMG/"
    img_prefix_center = "center"
    img_prefix_left = "left"
    img_prefix_right = "right"

    files = os.listdir(img_dir)

    total_files = len(files)
    total_files_13 = total_files / 3

    new_images = []
    new_wheels = []


    cnt = 0
    for file in files:
        if file.startswith("center"):

            if file in file_wheel_map:

                cnt += 1
                print(str(cnt) + "/" + str(total_files_13) + "   " + file)
                #img = mpimg.imread(img_dir + file)
                img = utils.read_image(img_dir + file)
                #np.append(new_images,img, axis=0)
                new_images.append(img)
                #new_images.add

                wheel = file_wheel_map[file]
                #new_wheels.append(wheel)
                new_wheels.append(wheel)

                #if cnt > 1000:
                #    break

                #print(type(img))
                print("image shape", img.shape)
                #print()
                #plt.imshow(img)
                #plt.show()

    new_images = np.array(new_images)
    new_wheels = np.array(new_wheels)

    #new_images = new_images.reshape(-1, 32*32*3)

    #print(len(new_images))
    #print(len(new_wheels))
    #print("type(new_images)", type(new_images))
    #print("type(new_wheels)", type(new_wheels))
    #print("type(new_images[0])", type(new_images[0]))
    #print("type(new_wheels)[0]", type(new_wheels[0]))

    new_images_train, new_images_test, new_wheels_train, new_wheels_test = train_test_split(np.array(new_images), new_wheels, test_size=0.20, random_state=42)

    train = {"features": new_images_train, "labels": new_wheels_train}
    pickle.dump(train,open("./train.p","wb"))

    test = {"features": new_images_test, "labels": new_wheels_test}
    pickle.dump(test,open("./test.p","wb"))



def load_train_vali_test():
    with open("./train.p","rb") as trainfile:
        train = pickle.load(trainfile)
    features_all = train["features"]
    labels_all = train["labels"]
    X_train1, X_validation1, y_train1, y_validation1 = train_test_split(features_all, labels_all, test_size=0.20, random_state=42)
    X_train1 = X_train1.astype('float32')
    X_validation1 = X_validation1.astype('float32')
    X_train1 = X_train1 / 255 - 0.5
    X_validation1 = X_validation1 / 255 - 0.5



    with open("./test.p","rb") as testfile:
        test = pickle.load(testfile)

    X_test1 = test["features"]
    y_test1 = test["labels"]
    X_test1 = X_test1.astype('float32')
    X_test1 = X_test1 / 255 - 0.5

    print(X_train1[0], y_train1[0], X_validation1[0], y_validation1[0], X_test1[0], y_test1[0])
    print(len(X_train1), len(X_validation1),len(X_test1), len(y_train1), len(y_validation1), len(y_test1))

    return X_train1, X_validation1, X_test1, y_train1, y_validation1, y_test1

def split_train_test_validate_file_names():
    file_wheel = utils.read_labels()
    files = list(file_wheel.keys())
    print(type(files))
    print(len(files))

    files_train, files_vali_test = train_test_split(files, test_size=0.07, random_state=42)
    files_vali, files_test = train_test_split(files_vali_test, test_size=0.5, random_state=42)

    print("train: ", len(files_train))
    print("vali: ", len(files_vali))
    print("test: ", len(files_test))

    return files_train, files_vali, files_test

def generate_train_test(files, file_wheel):

    #img_dir = data_dir + "IMG/"
    #img_prefix_center = "center"
    #img_prefix_left = "left"
    #img_prefix_right = "right"

    new_images = []
    new_wheels = []

    for file in files:
        file_path = file

        if os.path.exists(file_path)  and (file in file_wheel):
            img = utils.read_image(file_path)
            wheel = file_wheel[file]

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


def generate_train(files, file_wheel):

    batch_size = 128
    total_cnt = len(files)
    batch_num = int(total_cnt / batch_size)

    while 1:
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

            #print(len(new_images))
            #print(len(new_wheels))
            #print("type(new_images)", type(new_images))
            #print("type(new_wheels)", type(new_wheels))
            #print("type(new_images[0])", type(new_images[0]))
            #print("type(new_wheels)[0]", type(new_wheels[0]))

            yield (new_images, new_wheels)


def train_model():
    files_train, files_vali, files_test = split_train_test_validate_file_names()

    file_wheel = utils.read_labels()
    X_validation, Y_validation = generate_train_test(files_vali,file_wheel)
    X_test, Y_test = generate_train_test(files_test, file_wheel)


    #X_train, X_validation,X_test, y_train, y_validation, y_test = load_train_vali_test()

    #Y_train = np_utils.to_categorical(y_train, 11)
    #Y_validation = np_utils.to_categorical(y_validation, 11)
    #Y_test = np_utils.to_categorical(y_test, 11)

    #Y_train = y_train
    #Y_validation = y_validation
    #Y_test = y_test

    #X_train_flat = X_train.reshape(-1, 160*320*3)
    #X_validation_flat = X_validation.reshape(-1,160,320*30)

    batch_size = 128
    nb_classes = 360
    nb_epoch = 12
    img_rows, img_cols = 66, 200

    pool_size = (2,2)
    kernel_size = (3,3)

    #print(X_train.shape)
    #print(X_validation.shape)

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

    INIT = 'glorot_uniform'
    keep_prob = 0.2
    reg_val = 0.01
    ch, row, col = 3, 66, 200  # camera format

    model.add(Lambda(lambda x: x / 1.0 - 0.,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))

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

    model.compile(optimizer="adam", loss="mse")  # , metrics=['accuracy']



    #----------------------


    history = model.fit_generator(generate_train(files_train,file_wheel),128*200,5,verbose=1,validation_data=(X_validation,Y_validation),nb_val_samples=1000)

    model_j = model.to_json()
    with open("./model.json","w") as jf:
        json.dump(model_j,jf)
    #model.save("./model_save.model",True)

    model.save_weights("./model.h5",True)

    history2=model.evaluate(X_test, Y_test)
    print(history2)

train_model()










