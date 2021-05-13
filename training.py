import os
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
import cv2                
import numpy as np
import random

epochs = 1
img_size = 224
training_Data = []

Datadirectory = 'TrainDataset2000Images/'
Classes = ['Closed_Eyes', 'Open_Eyes']

def create_training_Data():
    for category in Classes:
        path = os.path.join(Datadirectory, category)
        class_num = Classes.index(category) # 0 or 1
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                new_array = cv2.resize(backtorgb, (img_size, img_size))
                training_Data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_Data()

random.shuffle(training_Data)

X = []
y = []

for features, label in training_Data:
    X.append(features)
    y.append(label)
    
X = np.array(X).reshape(-1, img_size, img_size, 3)
X = X / 255.0 
Y = np.array(y)

# Transfer learning

model = tf.keras.applications.mobilenet.MobileNet()
print(model.summary())
base_input = model.layers[0].input
base_output = model.layers[-4].output

flat_layer = layers.Flatten()(base_output)
final_output = layers.Dense(1)(flat_layer) 
final_output = layers.Activation('sigmoid')(final_output)
new_model = keras.Model(inputs = base_input, outputs = final_output)

new_model.compile(loss = 'binary_crossentropy',
                           optimizer = 'adam',
                           metrics = ['accuracy'])

new_model.fit(X, Y, epochs = epochs, validation_split = 0.1) 

new_model.save('my_model_'+str(epochs)+'_epochs.h5')

