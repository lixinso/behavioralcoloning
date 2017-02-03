import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import numpy as np
import cv2

#Preprocess the image. Resize and normalization
def preprocess(img):

    #Remove the edge parts, which is not so important
    img = img.crop((20, 140, 50, 270))
    #Resize to adapt NVidia model
    img = img.resize((200, 66))
    img1 = mpimg.pil_to_array(img)
    #img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2HSV)

    #print(type(img1))

    img1 = img1.astype('float32')
    img1 = img1 / 255 - 0.5

    return img1

#Read the image via Image library
def read_image(img_file_name):
    img = Image.open(img_file_name)
    img1 = preprocess(img)
    return img1

#Read a bunch of images
def read_images(img_file_names):
    imgs = []
    for img_file_name in img_file_names:
        img = read_image(img_file_name)
        imgs.append(img)

    imgs_arr = np.array(imgs)

    return imgs_arr



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

                    #Create a dictionary to keep the file--wheel mapping
                    file_wheel_map[center_file] = steering

    with open(new_label_file,"w") as nf:
        nf.write(new_label_text)

    return file_wheel_map



