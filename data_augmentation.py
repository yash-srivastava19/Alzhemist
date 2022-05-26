# !/usr/bin/python

""" Folder Key: 1=MildDemented """

""" Dataset Structure:
Alz_Dataset:
....test
..........1 (MildDemented)
.............1_001.jpeg
.............1_002.jpeg
..........2 (ModerateDemented)
.............1_001.jpeg
.............1_002.jpeg
..........3 (NonDemented)
.............1_001.jpeg
.............1_002.jpeg
..........4 (VeryMildDemented)
.............1_001.jpeg
.............1_002.jpeg

.
.
.

"""

# We want to be sure how much each parameter in Datagen varies. Anything outside the range will directly affect model performance.
import numpy 
from glob import glob
from dataclasses import dataclass
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img

@dataclass
class Params:
    COUNTER = 15
    rot_range = 180
    width_shift_range = 0.4
    height_shift_range = 0.4
    shear_range = 0.3
    zoom_range = 0.3
    h_flip = True
    v_flip = True


config = Params()

COUNTER = 15

DataGen = ImageDataGenerator(
    rotation_range = config.rot_range,
    width_shift_range = config.width_shift_range,
    height_shift_range = config.height_shift_range,
    shear_range = config.shear_range,
    zoom_range = config.zoom_range,
    horizontal_flip = config.h_flip,
    vertical_flip = config.v_flip,
    )


Debug : print("[INFO]: .......Augmenting the Dataset....")

for k in range(1,7):
    Debug : print(f"[INFO]: .......Glob Files for K = {k}....")
    imagelist = glob('Data/{}_Images/{}*'.format(k,k)) #Arrange all images in a class to a particular folder

    for eachImage in imagelist:
        Debug : print(f"[INFO]: .......Preprocessing Images in the Data folder for K = {k} ....... (Grayscale, Expanded Dims)")
        img = load_img(eachImage,color_mode = 'grayscale')
        x = img_to_array(img)
        x = numpy.expand_dims(x,0)


        i = 0
        Debug : print(f"[INFO]: .......Saving the Images in the Train Folder -  for K = {k} .......")
        for batch in DataGen.flow(x,batch_size = 1, save_to_dir = 'Train/{}_Images'.format(k),save_prefix = str(k),save_format = 'jpeg'):
            i+=1
            if i>COUNTER:
                break
        Debug : print(f"[INFO]: .......From a particular image, {COUNTER} images are augmented in the Train folder for K = {k} ....... ")

Debug : print(f"[INFO]: ....... Dataset Augmentation is completed. Now run  'Dataset.py' .........") 