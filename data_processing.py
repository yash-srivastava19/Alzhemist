# !/usr/bin/python

# INPUT PIPELINE :
from glob import glob
import tensorflow as tf
CLASSES = 5
IMG_SIZE = (50, 50)

""" Running this is actually really beneficial """

#  Check this:
train_ds = tf.keras.utils.image_dataset_from_directory(directory = "Alz_Dataset/train", colormode='grayscale', seed=312, label_mode='int', image_size=IMG_SIZE)
test_ds = tf.keras.utils.image_dataset_from_directory(directory = "Alz_Dataset/test", colormode='grayscale', seed=172, label_mode='int', image_size=IMG_SIZE)

