#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.models import Sequential
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


#Dimension de la imagen
img_width, img_height = 150, 150
#Carpeta que almacena las imagenes
#con estas se entrenara la red
print("[INFO] loading images...")
train_data_dir = 'data/train'
#carpeta con las muestras de validacion
validation_data_dir = 'data/validation'
#numero de imagenes que se concideran para la validacion
train_samples = 1500
#numero de images que se cocideran para la validacion
validation_samples = 800
INIT_LR = 1e-3
#numero de veces que se ejecutara las red sobre el conjunto de entrenamiento
#antes de empezar con la validacion

epoch = 10
#***** Inicio del modelo *****
model= Sequential()
#model.add(Convolution2D(nb_filter, nb_row, nb_col, ))
# nb_filter: Number of convolution filters to use.
# nb_row: Number of rows in the convolution kernel.
# nb_col: Number of columns in the convolution kernel.
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))



# ** FIn del modelo **
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / epoch)
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
# esta es la mejora de la configuración que utilizaremos para el entrenamiento
# en el que generamos un gran número de imágenes transformadas de manera que el
# modelo puede tratar con una gran variedad de escenarios del mundo real
train_datagen = ImageDataGenerator(
        rescale =1./255,
        shear_range =0.2,
        zoom_range =0.2,
        horizontal_flip =True)


# esta es la mejora de la configuración que utilizaremospara la prueba:
# sólo para reajuste
test_datagen = ImageDataGenerator(rescale=1./255)
# esta sección toma imágenes de la carpeta
# y las pasa al ImageGenerator que crea entonces
# un gran número de versiones transformadas
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

validation_generator =  test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

# aquí es donde se produce el proceso real
# y llevará algún tiempo ejecutar este paso.
print("[INFO] training network...")
history = model.fit_generator(
        train_generator,
        verbose=1,
        validation_data=validation_generator,
        samples_per_epoch=train_samples,
        epochs =epoch,
        validation_steps=validation_samples)
# for e in range(40):
#     score = model.evaluate(validation_generator, verbose=0)
#     print ('Test loss:', score[0])
#     print ('Test accuracy:', score[1])

# save the model to disk
print("[INFO] serializing network...")
model.save('soccer_balls.model')
print("[INFO] model saved...")

print("[INFO] saving model json...")
clssf = model.to_json()
with open("SoccerVali.json", "w") as json_file:
    json_file.write(clssf)
print("[INFO] saving weights h5...")
model.save_weights('BallsDweights.h5')

print(history.history.keys())

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
