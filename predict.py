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
from keras.layers import  Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, Conv2D
from keras.utils import np_utils
from keras.optimizers import Adam
from keras import backend as K

import json

import utils

from keras.models import model_from_json

#Open and load the saved model
with open("./model.json", "r") as jf:
    model_json = jf.read()
loaded_model = model_from_json(json.loads(model_json))
loaded_model.compile("adam", "mse")
loaded_model.load_weights("./model.h5")

#Read all the training material file names and wheel
file_wheel = utils.read_labels()

#Then select 1 of 1000 and do the prediction
ii = 0
for file in file_wheel:
    ii+=1
    if ii % 1000 != 0:
        continue

    image_arr = utils.read_images([file])
    #print(image_arr.shape)

    predicted = loaded_model.predict(image_arr,1,1)

    print("predicted      ", predicted)

