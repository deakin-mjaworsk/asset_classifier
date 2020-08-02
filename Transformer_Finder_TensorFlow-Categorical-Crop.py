#Python Package imports

!pip install cv2

import os
import numpy as np
import sys
import cv2 as cv
import csv
import os, fnmatch
import pandas as pd
from shutil import *
import subprocess
import numpy as np

#functions to classify images

def char_position(letter):
    return float(ord(letter) - 27)

def pos_to_char(pos):
    return chr(pos + 27)

char_position('B')

my_list = ['A', 'B', 'c'] 

new_string = [char_position(i) for i in my_list]

print(new_string)

!pip install Pillow

#set train and test folder locations

train_folder = "C:\Python\powercor_assets\\trainset\\"
test_folder = "C:\Python\powercor_assets\\testset\\"

import os
import numpy as np

print(train_folder)
os.chdir(train_folder)
train_files = os.listdir('.')

print("TRAINING FILES:")
print(train_files)

labels = []

for train in train_files:
    labels.append(train[0])
    
print(labels)
print(train_files)

train_labels = np.array(labels)

train_labels2 = [char_position(i) for i in train_labels]

train_labels3 = [pos_to_char(int(i)) for i in train_labels2]

print(train_labels2)
print(train_labels3)


train_images = []

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import mean,cov,cumsum,dot,linalg,size,flipud,argsort
from pylab import imread,subplot,imshow,title,gray,figure,show,NullLocator,imsave
from PIL import Image
from PIL import ImageFilter
    
#== Parameters =======================================================================
BLUR = 21
CANNY_THRESH_1 = 10
CANNY_THRESH_2 = 200
MASK_DILATE_ITER = 1
MASK_ERODE_ITER = 1
MASK_COLOR = (0.1,0.1,0.1) # In BGR format


#== Processing =======================================================================
for train in train_files:
        
#-- Read image -----------------------------------------------------------------------
    img = cv2.imread(train)
    #img = img1[1:151, 1:600]
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#-- Edge detection -------------------------------------------------------------------
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

#-- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    _, contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# Previously, for a previous version of cv2, this line was: 
#  contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# Thanks to notes from commenters, I've updated the code but left this note
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

#-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
# Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))

#-- Smooth mask, then blur it --------------------------------------------------------
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask
    
#-- Blend masked img into MASK_COLOR background --------------------------------------
    mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices, 
    img         = img.astype('float32') / 255.0                 #  for easy blending

    masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) # Blend
    masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit 

# split image into channels
    c_red, c_green, c_blue = cv2.split(img)

# merge with mask got on one of a previous steps
    img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0))

    img_b = img_a[1:100, 1:600]
    
    r = 100.0 / img_b.shape[1]
    dim = (100, int(img_b.shape[0] * r))
    resized = cv2.resize(img_b, dim, interpolation = cv2.INTER_AREA)
    
    #%matplotlib inline
    #plt.imshow(resized)
    #plt.show()
   
    train_images.append(resized)
	
train_images_np4 = np.array(train_images)
print(train_images_np4.shape)

#x = np.zeros( (28, 300, 600, 3) )
train_images_np = train_images_np4[:, :, :, 0]
print(train_images_np.shape)

trn_img = train_images_np.reshape(train_images_np.shape[1]*train_images_np.shape[2], train_images_np.shape[0])
trn_img = trn_img.transpose()
print(trn_img.shape)

!pip install keras

os.chdir(test_folder)
test_files = os.listdir('.')

print("TEST FILES:")
print(test_files)

test_images = []

import cv2
import numpy as np
    
#3. License Grant. Subject to the terms and conditions of this License, Licensor hereby grants You a worldwide, royalty-free, non-exclusive, perpetual (for the duration of the applicable copyright) license to exercise the rights in the Work as stated below:

#https://creativecommons.org/licenses/by-sa/3.0/legalcode
#http://answers.opencv.org/question/201115/removing-image-background-from-image-with-python/

#to Reproduce the Work, to incorporate the Work into one or more Collections, and to Reproduce the Work as incorporated in the Collections;
#to create and Reproduce Adaptations provided that any such Adaptation, including any translation in any medium, takes reasonable steps to clearly label, demarcate or otherwise #identify that changes were made to the original Work. For example, a translation could be marked "The original work was translated from English to Spanish," or a modification #could indicate "The original work has been modified.";
#to Distribute and Publicly Perform the Work including as incorporated in Collections; and,
#to Distribute and Publicly Perform Adaptations.
#For the avoidance of doubt:

#Non-waivable Compulsory License Schemes. In those jurisdictions in which the right to collect royalties through any statutory or compulsory licensing scheme cannot be waived, the #Licensor reserves the exclusive right to collect such royalties for any exercise by You of the rights granted under this License;
#Waivable Compulsory License Schemes. In those jurisdictions in which the right to collect royalties through any statutory or compulsory licensing scheme can be waived, the #Licensor waives the exclusive right to collect such royalties for any exercise by You of the rights granted under this License; and,
#Voluntary License Schemes. The Licensor waives the right to collect royalties, whether individually or, in the event that the Licensor is a member of a collecting society that #administers voluntary licensing schemes, via that society, from any exercise by You of the rights granted under this License.
#The above rights may be exercised in all media and formats whether now known or hereafter devised. The above rights include the right to make such modifications as are #technically necessary to exercise the rights in other media and formats. Subject to Section 8(f), all rights not expressly granted by Licensor are hereby reserved.	

	
#== Parameters =======================================================================
BLUR = 21
CANNY_THRESH_1 = 10
CANNY_THRESH_2 = 200
MASK_DILATE_ITER = 1
MASK_ERODE_ITER = 1
MASK_COLOR = (0.1,0.1,0.1) # In BGR format

#== Processing =======================================================================
for test in test_files:

#-- Read image -----------------------------------------------------------------------
    img = cv2.imread(test)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#-- Edge detection -------------------------------------------------------------------
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

#-- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    _, contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# Previously, for a previous version of cv2, this line was: 
#  contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# Thanks to notes from commenters, I've updated the code but left this note
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

#-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
# Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))

#-- Smooth mask, then blur it --------------------------------------------------------
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

#-- Blend masked img into MASK_COLOR background --------------------------------------
    mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices, 
    img         = img.astype('float32') / 255.0                 #  for easy blending

    masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) # Blend
    masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit 

# split image into channels
    c_red, c_green, c_blue = cv2.split(img)

# merge with mask got on one of a previous steps
    img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0))
    
    img_b = img_a[1:100, 1:600]
    
    r = 100.0 / img_b.shape[1]
    dim = (100, int(img_b.shape[0] * r))
    resized = cv2.resize(img_b, dim, interpolation = cv2.INTER_AREA)
    
    #%matplotlib inline
    #plt.imshow(resized)
    #plt.show()
   
    test_images.append(resized)
	
test_images_np4 = np.array(test_images)
print(test_images_np4.shape)

test_images_np = test_images_np4[:, :, :, 0]
print(test_images_np.shape)
tst_img = test_images_np.reshape(test_images_np.shape[1]*test_images_np.shape[2], test_images_np.shape[0])
tst_img = tst_img.transpose()
print(tst_img.shape)

import matplotlib.pyplot as plt
 
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.optimizers import adam

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(16, 100)),
    keras.layers.Dense(1600, activation=tf.nn.selu),
    keras.layers.Dense(60, activation=tf.nn.softmax)
])


epochs = 150
learning_rate = 0.1
decay_rate = learning_rate / epochs
momentum = 0.8
sgd = keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
#sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
opt = adam(lr=0.001, decay=1e-6)
#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#nadam=keras.optimizers.Nadam(lr=0.004, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
#model.compile(optimizer=sgd,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.compile(optimizer=tf.train.AdamOptimizer(),loss='sparse_categorical_crossentropy',metrics=["accuracy"])
model.fit(train_images_np,train_labels2,batch_size=218,epochs=221,verbose=1)

print("fit")
#clf2.fit(trn_img, train_labels)

test_labels = ['T', 'T', 'T', 'H', 'H', 'T', 'H', 'T', 'R', 'T', 'T', 'V', 'T', 'R', 'B', 'H', 'B', 'H', 'R', 'R', 'R', 'R', 'T', 'B', 'T', 'R', 'H', 'T', 'B', 'T', 'B', 'H', 'B', 'B', 'H', 'B', 'B', 'B', 'B', 'B', 'B', 'R', 'T', 'B', 'B', 'H', 'T', 'R', 'B', 'B', 'T', 'H', 'T', 'B', 'V', 'T', 'B', 'R', 'T', 'B', 'B', 'B', 'R', 'B', 'B', 'B', 'B', 'T', 'T', 'H', 'T', 'T', 'B', 'B', 'B', 'R', 'B', 'R', 'H', 'B', 'B', 'B', 'T', 'B', 'R', 'B', 'B', 'B', 'T', 'R', 'B', 'T', 'B', 'B', 'B', 'B', 'T', 'B', 'T', 'H', 'B', 'T', 'T', 'T']
test_labels2 = [char_position(i) for i in test_labels]

# evaluate the model
scores = model.evaluate(train_images_np, train_labels2)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

test_loss, test_acc = model.evaluate(test_images_np, test_labels2)
print('Test accuracy:', test_acc)

y_pred = model.predict(test_images_np,verbose=1)
print(y_pred)


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

test_labels = ['T', 'T', 'T', 'H', 'H', 'T', 'H', 'T', 'R', 'H', 'T', 'V', 'T', 'R', 'B', 'H', 'B', 'H', 'R', 'R', 'R', 'R', 'T', 'B', 'T', 'R', 'H', 'T', 'B', 'T', 'B', 'H', 'B', 'B', 'H', 'B', 'B', 'B', 'B', 'B', 'B', 'R', 'T', 'B', 'B', 'H', 'T', 'R', 'B', 'B', 'T', 'H', 'T', 'B', 'V', 'T', 'B', 'R', 'T', 'B', 'B', 'B', 'R', 'B', 'B', 'B', 'B', 'T', 'T', 'H', 'T', 'T', 'B', 'B', 'B', 'R', 'B', 'R', 'H', 'B', 'B', 'B', 'T', 'B', 'R', 'B', 'B', 'B', 'T', 'R', 'B', 'T', 'B', 'B', 'B', 'B', 'T', 'B', 'T', 'H', 'B', 'T', 'T', 'T']
class_names = [ 'B', 'F', 'G', 'H', 'R', 'T', 'V']


ypred2 = [pos_to_char(int(np.argmax(i))) for i in y_pred]
print(ypred2)

for i in range(len(test_files)):
    if str(ypred2[i]) == 'T':
        plt.imshow(test_images[i])
        plt.show()

report = classification_report(ypred2, test_labels)
print(report)

from sklearn.metrics import accuracy_score

acc = accuracy_score(test_labels, ypred2)
print(acc)

results_folder = "C:\Python\powercor_assets\\resultset\\"
os.chdir(results_folder)

with open("results.txt","w+") as f:
    
    for i in range(len(test_files)):
        if str(ypred2[i]) == 'B':
            f.write(str.replace(test_files[i], ".jpg", "") + " = Bush \n"))
        if str(ypred2[i]) == 'G':
            f.write(str.replace(test_files[i], ".jpg", "") + " = Grass \n"))
        if str(ypred2[i]) == 'H':
            f.write(str.replace(test_files[i], ".jpg", "") + " = House \n"))
        if str(ypred2[i]) == 'R':
            f.write(str.replace(test_files[i], ".jpg", "") + " = Road \n"))
        if str(ypred2[i]) == 'T':
            f.write(str.replace(test_files[i], ".jpg", "") + " = Transformer \n"))
        if str(ypred2[i]) == 'V':
            f.write(str.replace(test_files[i], ".jpg", "") + " = Vehicle \n"))
    
f.close()

