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



