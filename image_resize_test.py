import PIL
from PIL import Image
import os

def resize_file(file_name):

    im = Image.open("../data/IMG/" + file_name)
    #im.thumbnail((80,160),Image.ANTIALIAS)
    img1 = im.resize((200, 66))
    img1.save("../data/IMG_resized_nvidia/" + file_name)

files = os.listdir("../data/IMG/")
for file in files:
    resize_file(file)
