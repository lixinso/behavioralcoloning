import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import numpy as np

def preprocess(img):
    img = img.resize((200, 66))
    img1 = mpimg.pil_to_array(img)
    #print(type(img1))

    img1 = img1.astype('float32')
    img1 = img1 / 255 - 0.5

    return img1

def read_image(img_file_name):
    img = Image.open(img_file_name)
    img1 = preprocess(img)
    return img1

def read_images(img_file_names):
    imgs = []
    for img_file_name in img_file_names:
        img = read_image(img_file_name)
        imgs.append(img)

    imgs_arr = np.array(imgs)

    return imgs_arr



data_dir = "../session_data/"
def read_labels():
    file_wheel_map = {}

    label_dir = data_dir + "driving_log.csv"

    new_label_file = data_dir + "new_driving_log.csv"
    new_label_text = ""

    cnt = 0
    with open(label_dir) as lf:
        for line in lf:

            cnt += 1

            if cnt  == 1:
                continue

            columns = line.strip().split(",")
            if len(columns) > 4:
                center_file = columns[0].strip()#[4:]
                left_file = columns[1].strip()#[4:]
                right_file = columns[2].strip()#[4:]
                #print(columns[3])
                steering = float(columns[3].strip()) #int( round( float(columns[3].strip()) * 5.0 ) )

                print(center_file, steering)
                new_label_text += center_file + "," + str(steering) + "\n"

                file_wheel_map[center_file] = steering

    with open(new_label_file,"w") as nf:
        nf.write(new_label_text)

    return file_wheel_map




