#Interface to run CW/EAD attacks

from re import T
import sys,os

from numpy.core.fromnumeric import argmax
BASE_DIR = os.path.dirname(os.path.abspath('/home/ronghui/cnncertify'))
import subprocess
import numpy as np
#from cnn_bounds_full import run as run_cnn_full
#from cnn_bounds_full_core import run as run_cnn_full_core

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dot, Input, Dense, Activation, Flatten, Lambda, Conv2D,Conv1D, Add, AveragePooling2D, BatchNormalization, Lambda,GlobalMaxPooling1D,Reshape,Dropout,GlobalAveragePooling1D,MaxPool1D,AveragePooling1D
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
#from keras.backend import manual_variable_initialization
#manual_variable_initialization(True)
from Attack.li_attack import CarliniLi
from Attack.l2_attack import CarliniL2
from Attack.l1_attack_new import EADL1
from Attack.attack2 import attack_2
from Attack.attack_1 import attack_1
import tensorflow as tf
import time as timer

import datetime
from utils import generate_pointnet_data

f = open('./cw.txt', "a+")
def loss(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted)

#Runs CW/EAD attack with specified norm
def cw_attack( sess,model,norm, num_image=10, cifar = False, tinyimagenet = False):
    
    if norm == '1':
        attack = EADL1
        norm_fn = lambda x: np.sum(np.abs(x),axis=(1,2))
    elif norm == '2':
        attack = CarliniL2
        norm_fn = lambda x: np.sum(x**2,axis=(1,2))
    elif norm == 'i':
        attack = CarliniLi
        norm_fn = lambda x: np.max(np.abs(x),axis=(1,2))
        
        

    
    
    NUM_POINTS = 512
    NUM_CLASSES = 40
    BATCH_SIZE = 32
    def conv_bn(x, filters):
        x = Conv1D(filters, kernel_size=1, padding="valid")(x)
        x = BatchNormalization(momentum=0.0)(x)
        return Lambda(tf.nn.relu)(x)


    def dense_bn(x, filters):
        x = Dense(filters)(x)
        x = BatchNormalization(momentum=0.0)(x)
        return Lambda(tf.nn.relu)(x)
    class OrthogonalRegularizer(Regularizer):
        def __init__(self, num_features, l2reg=0.001):
            self.num_features = num_features
            self.l2reg = l2reg
            self.eye = tf.eye(num_features)

        def __call__(self, x):
            x = tf.reshape(x, (-1, self.num_features, self.num_features))
            xxt = tf.tensordot(x, x, axes=(2, 2))
            xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
            return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

    def tnet(inputs, num_features):

        # Initalise bias as the indentity matrix
        bias = Constant(np.eye(num_features).flatten())
        reg = OrthogonalRegularizer(num_features)

        x = conv_bn(inputs, 32)
        x = conv_bn(x, 64)
        x = conv_bn(x, 512)
        #x = GlobalAveragePooling1D()(x)
        x = GlobalMaxPooling1D()(x)
        x = dense_bn(x, 256)
        x = dense_bn(x, 128)
    

        x = Dense(
            num_features * num_features,
            kernel_initializer="zeros",
            bias_initializer=bias,
            activity_regularizer=reg,
        )(x)
    
        #x = Reshape((1, 9))(x)
        feat_T = Reshape((num_features, num_features))(x)
        # Apply affine transformation to input features
        #return x
        return Dot(axes=(2, 1))([inputs, feat_T])
    
    inputs = Input(shape=(NUM_POINTS, 3))
    
    
    
    
    x = tnet(inputs, 3)
    x = conv_bn(x, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    #x = GlobalAveragePooling1D()(x)
    x = GlobalMaxPooling1D()(x)
    x = dense_bn(x, 512)
    x = Dense(256)(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization(momentum=0.0)(x)
    x = Lambda(tf.nn.relu)(x)
    outputs = Dense(NUM_CLASSES)(x)

    #outputs = Activation('linear')(x)
    model = Model(inputs=inputs, outputs=outputs, name="pointnet")
    model.summary()
    #model.load_weights('./pretrained40_weights_average_64_20.h5')
    
    
    '''
      #'./pretrained40_weights_average_64_25.h5' 51%
    
    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 64)
    #x = conv_bn(x, 512)
    x = GlobalAveragePooling1D()(x)
    x = dense_bn(x, 512)
    x = dense_bn(x, 128)
    x = Dense(9)(x)
    feat_T = Reshape((3,3))(x)
    x = Dot(axes=(2, 1))([inputs, feat_T])
    x = conv_bn(x, 32)
    #x = conv_bn(x, 32)
    x = conv_bn(x, 512)
    x = conv_bn(x, 512)
    x = conv_bn(x, 1024)
    x = GlobalAveragePooling1D()(x)
    x = dense_bn(x, 512)
    x = Dense(256)(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization(momentum=0.0)(x)
    x = Lambda(tf.nn.relu)(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES)(x)
    model = Model(inputs=inputs, outputs=outputs, name="pointnet")
    model.summary()

    model.load_weights('./pretrained40_weights_average_64_25.h5')
    
    
    '''
  
    
    '''
    
    x = tnet(inputs, 3)
    x = conv_bn(x, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = conv_bn(x, 1024)
    x = GlobalAveragePooling1D()(x)
    x = dense_bn(x,1024)
    x = dense_bn(x, 512)
    x = dense_bn(x, 512)
    x = Dropout(0.3)(x)
    x = dense_bn(x, 128)
    
    '''
    '''
    #./pretrained40_weights_average_64_9.h5

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 9)
    x = GlobalMaxPooling1D()(x)
    feat_T = Reshape((3,3))(x)
    x = Dot(axes=(2, 1))([inputs, feat_T])
    x = conv_bn(x, 32)
    x = conv_bn(x, 64)
    x = GlobalMaxPooling1D()(x)
    x = dense_bn(x, 128)
    outputs = Dense(NUM_CLASSES)(x)
    model = Model(inputs=inputs, outputs=outputs, name="pointnet")
    model.summary()
    
    
    '''
     

    #model.load_weights('./pretrained40_weights_average_64_30.h5')
    model.load_weights('./pretrained40_weights_max_512_20.h5')
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(lr=0.001),
        metrics=["sparse_categorical_accuracy"],
    )
    
    
   
    
    
    
    
    inputs, targets, true_labels, true_ids, img_info = generate_pointnet_data(NUM_POINTS, targeted=True,
                                                                              random_and_least_likely=True,
                                                                              target_type=0b0001,
                                                                              predictor=model.predict, start=0)
    if cifar:
        model.image_size = 32
        model.num_channels = 3
    elif tinyimagenet:
        model.image_size = 64
        model.num_channels = 3
        model.num_labels = 200
    else:
        model.image_size = 64
        model.num_channels = 3
        
    
    start_time = timer.time()
    attack = attack(sess, model, max_iterations = 1000)
    #tau = attack.pgd_attack(inputs, targets)
    tau= attack.attack(inputs, targets)
    #dist= attack.attack(inputs, targets)
    #perturbed_input= attack.attack(inputs, targets)
    
    #perturbed_input,dist = attack_2(sess, model,inputs, targets)
    
    #UB = np.average(norm_fn(perturbed_input-inputs[0]))
    #UB = np.average(norm_fn(perturbed_input))
    #avescore = np.average(dist)
    
    UB = np.average(tau)
    #print('f_tau is',tau)
    #print('f_tau is li',tau,file =f)
    
    #print('UB l1 is',UB)
    
    #print('UB l1 is',UB,file =f)
    #print('dist l1 is',avescore)
    #print('UB l1 is',avescore,file =f)
    print('UB is',UB)
    #print('time is',(timer.time()-start_time)/len(inputs))
    #print('UB len is',len(dist),file =f)
    #print('UB len is',len(dist))
    return UB, (timer.time()-start_time)/len(inputs)
    
