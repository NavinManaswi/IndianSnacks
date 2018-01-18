import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm
from scipy import misc
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt
import indiansnacksdataset as data
tf.reset_default_graph()
TRAIN_DIR ='train_images'
TEST_DIR ='test_images'
IMG_SIZE = 280
LR = 1e-3
#MODEL_NAME = 'quickest.model'.format(LR, '2conv-basic')


train_data= data.load_train_data()
train = train_data[:-200]
test = train_data[-200:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]
test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]
X = X.reshape([-1, 280, 280, 1])
test_x = test_x.reshape([-1, 280, 280, 1])

convnet = input_data(shape=[None, 280, 280, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 10, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')


model = tflearn.DNN(convnet)




model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}) , show_metric=True)


test_data = data.load_test_data()

fig=plt.figure()

for num,data in enumerate(test_data[:12]):

    
    img_num = data[1]
    img_data = data[0]
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    #model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 0: str_label='samosa'
    elif np.argmax(model_out) == 1: str_label='kachori'
    elif np.argmax(model_out) == 2: str_label='aloo_paratha'
    elif np.argmax(model_out) == 3: str_label='idli'
    elif np.argmax(model_out) == 4: str_label='jalebi'
    elif np.argmax(model_out) == 5: str_label='tea'
    elif np.argmax(model_out) == 6: str_label='paneer_tikka'
    elif np.argmax(model_out) == 7: str_label='dosa'
    elif np.argmax(model_out) == 8: str_label='omlet'
    elif np.argmax(model_out) == 9: str_label='poha'
        
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()


