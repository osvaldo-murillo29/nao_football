
from keras.models import model_from_json
from keras.preprocessing import image
#from cnn.py import training_set
import matplotlib.pyplot as plt
from matplotlib import ticker
import cv2
import numpy as np


json_file = open('SoccerVali.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("BallsDweights.h5")
print("Loaded model from disk")

'''loaded_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
'''
loaded_model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

ball = raw_input("Name of the ball: ")

img_pred = image.load_img('/home/mr-robot/Documents/nao_soccer/single_prediction/'+ball+'.jpg', target_size = (150, 150))
plt.figure(figsize=(10,5))
plt.imshow(img_pred)
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)
rslt = loaded_model.predict(img_pred)
#
#ind = training_set.class_indices

if rslt[0][0] == 1:
    prediction = "soccer"
    #print("Creo que es un perro")
    print('I am sure this is a Soccer Ball')
else:
    prediction = "baseball"
    print('I am sure this is a Baseball Ball')


plt.show()
