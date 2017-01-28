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

import utils

from keras.models import model_from_json

#image_arr = utils.read_images(["/Users/lixinso/Desktop/udacity_sdc/term1/project_behavioralcloning/data/IMG/center_2016_12_01_13_43_17_948.jpg"])
#image_arr = utils.read_images(["/Users/lixinso/Desktop/udacity_sdc/term1/project_behavioralcloning/data/IMG/center_2016_12_01_13_45_27_897.jpg"])
image_arr = utils.read_images(["/Users/lixinso/Desktop/udacity_sdc/term1/project_behavioralcloning/data/IMG/center_2016_12_01_13_33_26_862.jpg"])

with open("./model.json","r") as jf:
    model_json = jf.read()

loaded_model = model_from_json(model_json)
loaded_model.load_weights("./model.h5")

print(image_arr.shape)

predicted = loaded_model.predict(image_arr,1,1)

print(predicted)


