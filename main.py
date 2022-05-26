"""
Main CNN-Cert interfacing file

Copyright (C) 2018, Akhilan Boopathy <akhilan@mit.edu>
                    Lily Weng  <twweng@mit.edu>
                    Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
                    Sijia Liu <Sijia.Liu@ibm.com>
                    Luca Daniel <dluca@mit.edu>
"""
import subprocess
import numpy as np
from cnn_bounds_full import run as run_cnn_full
from cnn_bounds_full_core import run as run_cnn_full_core
from Attack.cw_attack import cw_attack
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dot, Input, Dense, Activation, Flatten, Lambda, Conv2D,Conv1D, Add, AveragePooling2D, BatchNormalization, Lambda,GlobalMaxPooling1D,Reshape,Dropout,GlobalAveragePooling1D,MaxPooling1D,ReLU
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
#from keras.backend import manual_variable_initialization
#manual_variable_initialization(True)
import tensorflow as tf
import time as timing
import datetime
from utils import generate_pointnet_data
ts = timing.time()
timestr = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')

#Prints to log file
def printlog(s):
    print(s, file=open("log_pymain_"+timestr+".txt", "a"))
    
#Runs command line command
def command(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')

#Runs Fast-Lin with specified parameters
def run(hidden, numlayer, numimage, norm, filename = '', layers = None, lp=False, lpfull= False, dual=False, sparse = False, spectral = False, cifar = False, cnnmodel = False, tinyimagenet=False):
    if sparse:
        cmd = 'python3 Fast-Lin/main_sparse.py '
    else:
        cmd = 'python3 Fast-Lin/main.py '
    if cifar:
        cmd += '--model cifar '
    if tinyimagenet:
        cmd += '--model tiny '
    if spectral:
        cmd += '--method spectral '
    if cnnmodel:
        cmd += '--cnnmodel '
    cmd += '--hidden ' + str(hidden) + ' '
    cmd += '--numlayer ' + str(numlayer) + ' '
    cmd += '--numimage ' + str(numimage) + ' '
    cmd += '--norm ' + str(norm) + ' '
    if lp:
        cmd += '--LP '
    if lpfull:
        cmd += '--LPFULL '
    if dual:
        cmd += '--dual '
    if filename:
        cmd += '--filename ' + str(filename) + ' '
        cmd += '--layers ' + ' '.join(str(l) for l in layers) + ' '
    cmd += '--eps 0.05 --warmup --targettype random'
    printlog("cmd: " +str(cmd))
    result = command(cmd)
    result = result.rsplit('\n',2)[-2].split(',')
    LB = result[1].strip()[20:]
    time = result[3].strip()[17:]
    return float(LB), float(time)
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
    #bias = Constant(np.eye(num_features).flatten())
    #reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    #x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = GlobalAveragePooling1D()(x)
    
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = Dense(
        num_features * num_features
       # kernel_initializer="zeros",
       # bias_initializer=bias,
       # activity_regularizer=reg,
    )(x)
    
    #x = Flatten()(x)
    feat_T = Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    
    #return x
    return Dot(axes=(2, 1))([inputs, feat_T])
inputs = Input(shape=(NUM_POINTS, 3))
#x = conv_bn(inputs, 32)
'''
x = tnet(inputs, 3)

#x = conv_bn(inputs, 32)
#x = conv_bn(x, 64)
#x = conv_bn(x, 9)
#x = GlobalAveragePooling1D()(x)

#x =Reshape((1,9))(x)
#feat_T = Reshape((3,3))(x)
#x = Dot(axes=(2, 1))([inputs, feat_T])
#x = tnet(x, 32)
#x = conv_bn(x, 32)
x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 512)

x = GlobalAveragePooling1D()(x)
#x = GlobalMaxPooling1D()(x)
x = dense_bn(x, 256)
x = Dropout(0.3)(x)
x = dense_bn(x, 128)
x = Dropout(0.3)(x)
'''
'''
# ./pretrained40_weights_average_64_14.h5
x = conv_bn(inputs, 64)
x = conv_bn(x, 64)
x = conv_bn(x, 64)
x = conv_bn(x, 128)
x = conv_bn(x, 128)
x = GlobalAveragePooling1D()(x)
x = dense_bn(x, 512)
x = Dense(256)(x)
x = Dropout(0.3)(x)
x = BatchNormalization(momentum=0.0)(x)
x = Lambda(tf.nn.relu)(x)
'''

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
'''

'''
#'./pretrained40_weights_average_64_20.h5'
x = tnet(inputs, 3)
x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 512)
x = GlobalMaxPooling1D()(x)
x = dense_bn(x, 512)
x = Dense(256)(x)
x = Dropout(0.3)(x)
x = BatchNormalization(momentum=0.0)(x)
x = Lambda(tf.nn.relu)(x)

'''



#'./pretrained40_weights_average_64_20.h5'
x = tnet(inputs, 3)
x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 512)
x = GlobalAveragePooling1D()(x)
x = dense_bn(x, 512)
x = Dense(256)(x)
x = Dropout(0.3)(x)
x = BatchNormalization(momentum=0.0)(x)
x = Lambda(tf.nn.relu)(x)




'''

#'./pretrained40_weights_average_64_30.h5'
x = tnet(inputs, 3)
x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 512)
x = conv_bn(x, 1024)
x = GlobalMaxPooling1D()(x)
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

'''





outputs = Dense(NUM_CLASSES)(x)

#outputs = Activation('linear')(x)
model = Model(inputs=inputs, outputs=outputs, name="pointnet")
model.summary()

#model.load_weights('./pretrained40_weights_average_64_30.h5')
#model.load_weights('./pretrained40_weights_ave_512_20.h5')
#model.load_weights('./pretrained40_weights_average_64_9.h5')
#model.load_weights('./pretrained40_weights_average_64_14.h5')
model.load_weights('./pretrained40_weights_average_64_20.h5')
#model.load_weights('./pretrained40_weights_average_64_25.h5')
#model.load_weights('./pretrained_weights_noet_average_512.h5')

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(lr=0.001),
    metrics=["sparse_categorical_accuracy"],
)
#Runs CNN-Cert with specified parameters
def run_cnn(model, inputs, targets, true_labels, true_ids, img_info, n_samples, norm, activation='relu'):
        if norm == 'i':
            return run_cnn_full(model, inputs, targets, true_labels, true_ids, img_info,  n_samples, 105, 1, activation, cifar, tinyimagenet)
        elif norm == '2':
            return run_cnn_full(model, inputs, targets, true_labels, true_ids, img_info,  n_samples, 2, 2, activation, cifar, tinyimagenet)
        if norm == '1':
            return run_cnn_full(model, inputs, targets, true_labels, true_ids, img_info,  n_samples, 1, 105, activation, cifar, tinyimagenet)

#Runs all Fast-Lin and CNN-Cert variations
def run_all_relu(model,inputs, targets, true_labels, true_ids, img_info,  cifar = False, num_image=10, flfull = False, nonada = True):

    filters = None
    LBs = []
    times = []
    #for norm in ['i']:
    for norm in ['2']:
        LBss = []
        timess = []
        if nonada: #Run non adaptive CNN-Cert bounds
            LB, time = run_cnn(model,inputs, targets, true_labels, true_ids, img_info, num_image, norm)
            printlog("CNN-Cert-relu")
  
            printlog("avg robustness = {:.5f}".format(LB))
            printlog("avg run time = {:.2f}".format(time)+" sec")
            printlog("-----------------------------------")
            LBss.append(LB)
            timess.append(time)
        
    return LBs, times




if __name__ == '__main__':
    LB = []
    
    prob_predict = model.predict
    
    inputs, targets, true_labels, true_ids, img_info = generate_pointnet_data(NUM_POINTS, targeted=True, random_and_least_likely = True, target_type = 0b0001, predictor=model.predict, start=0) #top2
    #inputs, targets, true_labels, true_ids, img_info = generate_pointnet_data(NUM_POINTS,targeted=True, random_and_least_likely = True, target_type = 0b10000, predictor=model.predict, start=0) #full
    #print("[DATAGEN][L1] no = {}, true_id = {}, true_label = {}, predicted = {}, correct = {}, seq = {}, info = {}".format(total, start + i,
     #                   test_labels[start+i], predicted_label, test_labels[start+i]== predicted_label, seq, [] if len(seq) == 0 else information[-len(seq):]))
    table = 0
    print("==================================================")
    print("================ Running Table {} ================".format(table))
    print("==================================================")
    
    
    printlog("-----------------------------------")
    
    LBs, times = run_all_relu(model, inputs, targets, true_labels, true_ids, img_info)
   
