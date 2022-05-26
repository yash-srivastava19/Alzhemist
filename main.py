
import os
import time
import numpy
import subprocess
import tensorflow as tf
import keras_tuner as kt
from dataclasses import dataclass

# Set Environment Variables.
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"

# We need to see what dimensions we need for our images. We want to be able to pass it through pipelines.
@dataclass
class Hyperparameters:
    CLASSES = 7   # See this ...........
    INPUT_SHAPE = (28,28,1)
    BATCH_SIZE = 5
    EPOCHS = 100
    LOGS_DIR = "/tmp/tb/tf_logs/Dice_MNIST/" + time.strftime('%d-%m-%Y_%H-%M-%S') 
    IMG_SIZE = (50, 50)

config = Hyperparameters()

#Add your path to the numpy files here
print("[INFO]: .......... Preprocessing Datasets ..........")

print("[INFO]: .......... Loading Test and Train Datasets ..........")

train_ds = tf.keras.utils.image_dataset_from_directory(directory = "Alz_Dataset/train", colormode='grayscale', seed=312, label_mode='int', image_size=config.IMG_SIZE)
test_ds = tf.keras.utils.image_dataset_from_directory(directory = "Alz_Dataset/test", colormode='grayscale', seed=172, label_mode='int', image_size=config.IMG_SIZE)

# It is recommended to double check these hyperparameters by tuning(preferably by using kerastuner)

print("[INFO]: .......... Creating Model ..........")
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape= config.INPUT_SHAPE,name = "InputLayer"),
    
    tf.keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu',padding='same',name = "Conv1"),
    tf.keras.layers.MaxPooling2D(padding='same',name = "MaxPool1"),
                           
    tf.keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu',padding='same',name = "Conv2"),
    tf.keras.layers.MaxPooling2D(padding='same',name = "MaxPool2"),                   
    
    tf.keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu',padding='same',name = "Conv3"),
    tf.keras.layers.MaxPooling2D(padding='same',name = "MaxPool3"),                       
                           
    tf.keras.layers.Flatten(name = "FlatLayer1"),
    tf.keras.layers.Dropout(0.1,name = "Dropout1"),
                           
    tf.keras.layers.Dense(60,name = "Dense60"),                      
    tf.keras.layers.Dense(config.CLASSES,activation='softmax',name = "Dense6")
])

Debug: model.summary()
Save : tf.keras.utils.plot_model(model,to_file = 'model.png',show_dtype = False, show_shapes = False, show_layer_names = True)
    
tBoardCallback = tf.keras.callbacks.TensorBoard(config.LOGS_DIR,histogram_freq = 1, profile_batch = (500,520))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

MLambda = lambda x : model

print('[INFO]: ........ Hyperparameter tuning - Keras Tuner ...........')

tuner = kt.RandomSearch(MLambda, objective='accuracy', max_trials=5)
tuner.search(train_ds, epochs=5, validation_data=test_ds)
model = tuner.get_best_models()[0]

print('[INFO]: ........ Hyperparameter tuning Completed ...........')

print("[INFO]: .......... Starting Model Training ..........")
model.fit(train_ds,batch_size=config.BATCH_SIZE ,epochs=config.EPOCHS,callbacks = [tBoardCallback])

# print("[INFO]: .......... Evaluating the Model ..........")

# numpy.seterr(divide = 'ignore')
loss,acc = model.evaluate(test_ds)

# Debug: print("Loss:{}  Accuracy:{}".format(loss,acc))
# Save : model.save("DiceMNIST.h5")


print("[INFO]: ..............Done with Training. Opening Tensorboard ..................")
subprocess.Popen(["tensorboard", "--logdir", "/tmp/tb/tf_logs/Dice_MNIST/"])