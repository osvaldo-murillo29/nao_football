import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.callbacks import Callback, ModelCheckpoint

img_width, img_height = 150, 150
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
model_weights_file = 'soccer_balls.h5'
train_samples = 2000
validation_samples = 800
epoach = 50

#Dimension of the Image
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

train_samples = 2000
validation_samples = 800

epoch =50

model = Sequential()

model.add(Convolution2D(32,(3,3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
(
model.add(Convolution2D(32,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
