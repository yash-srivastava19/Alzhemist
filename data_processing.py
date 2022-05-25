# !/usr/bin/python

# INPUT PIPELINE :
from glob import glob
import tensorflow as tf
CLASSES = 5

""" Running this is actually really beneficial """

#  Check this:
train_ds = tf.keras.utils.image_dataset_from_directory("Alz_Dataset/train", colormode='grayscale' seed=312, label_mode='int')
print(train_ds.class_names)


test_ds = tf.keras.utils.image_dataset_from_directory("Alz_Dataset/test", colormode='grayscale' seed=172, label_mode='int')
print(test_ds.class_names)

Debug: print(
    "[INFO]: ......... Preparing Train Dataset and saving as NumPy Array .......")

print("[INFO]: ......... Saved Train Dataset ........... ")


print("[INFO]: ......... Saved Test Dataset ........... ")

print("[INFO]: ......... Done with Preparing the Dataset - Now run 'SpiceyDicey.py' ........... ")
