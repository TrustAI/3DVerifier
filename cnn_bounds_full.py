"""
cnn_bounds_full.py

BAsed on Main CNN-Cert computation file for general networks

Copyright (C) 2018, Akhilan Boopathy <akhilan@mit.edu>
                    Lily Weng  <twweng@mit.edu>
                    Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
                    Sijia Liu <Sijia.Liu@ibm.com>
                    Luca Daniel <dluca@mit.edu>
Modified by Ronghui Mu <ronghui.mu@lancaster.ac.uk>
"""
from numba import njit, jit
import numpy as np
import time

from numpy.lib.function_base import average
from cnn_bounds_full_core import pool, conv, conv_bound, conv_full, conv_bound_full, pool_linear_bounds
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dot,Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, Lambda
from tensorflow.contrib.keras.api.keras.layers import Conv2D, GlobalMaxPooling1D, MaxPooling2D, AveragePooling2D, InputLayer, BatchNormalization, Reshape, GlobalAveragePooling1D, Conv1D
from tensorflow.contrib.keras.api.keras.models import load_model
from tensorflow.contrib.keras.api.keras import backend as K
from train_resnet import ResidualStart, ResidualStart2
import tensorflow as tf
from utils import generate_pointnet_data
import time
from joblib import Parallel, delayed
from activations import relu_linear_bounds, ada_linear_bounds, atan_linear_bounds, sigmoid_linear_bounds, tanh_linear_bounds
linear_bounds = None
import random

def loss(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                    logits=predicted)
#General model class
#f = open('./very-3d-test.txt', "a+")
num_points = 512
repeat = False
class Model:
    def __init__(self, model, inp_shape = (num_points,3)):
        temp_weights = [layer.get_weights() for layer in model.layers]
        self.shapes = []
        self.sizes = []
        self.weights = []
        self.biases = []
        self.pads = []
        self.strides = []
        self.types = []
        self.model = model
        
        cur_shape = inp_shape
        self.shapes.append(cur_shape)
        i = 0
        while i < len(model.layers):
            layer = model.layers[i]
            i += 1
            print(cur_shape)
            weights = layer.get_weights()
            if type(layer) == Conv1D:
                print('conv')
                if len(weights) == 1:
                    W = weights[0].astype(np.float32)
                    b = np.zeros(W.shape[-1], dtype=np.float32)
                else:
                    W, b = weights
                    W = W.astype(np.float32)
                    b = b.astype(np.float32)
                padding = layer.get_config()['padding']
                stride = layer.get_config()['strides']
                pad = (0,0,0,0) #p_hl, p_hr, p_wl, p_wr
                if padding == 'same':
                    desired_h = int(np.ceil(cur_shape[0]/stride[0]))
                    desired_w = int(np.ceil(cur_shape[0]/stride[1]))
                    total_padding_h = stride[0]*(desired_h-1)+W.shape[0]-cur_shape[0]
                    total_padding_w = stride[1]*(desired_w-1)+W.shape[1]-cur_shape[1]
                    pad = (int(np.floor(total_padding_h/2)),int(np.ceil(total_padding_h/2)),int(np.floor(total_padding_w/2)),int(np.ceil(total_padding_w/2)))
                cur_shape = (int((cur_shape[0]+pad[0]+pad[1]-W.shape[0])/stride[0])+1, W.shape[-1])
                W = np.ascontiguousarray(W.transpose((2,0,1)).astype(np.float32))
                b = np.ascontiguousarray(b.astype(np.float32))
                self.types.append('conv')
                self.sizes.append(None)
                self.strides.append(stride)
                self.pads.append(pad)
                self.shapes.append(cur_shape)
                self.weights.append(W)
                self.biases.append(b)
      
           
            elif type(layer) == GlobalAveragePooling1D:
                print('global avg pool')
                b = np.zeros(cur_shape[-1], dtype=np.float32)
                W = np.zeros((cur_shape[1],cur_shape[0],cur_shape[1]), dtype=np.float32)
                
                for f in range(W.shape[0]):
                    W[f,:,f] = 1/(cur_shape[0])
                #W=csr_matrix(W)
                pad = (0,0,0,0) #p_hl, p_hr, p_wl, p_wr
                #pool_size = layer.get_config()['pool_size']
                stride = (1,)
                #cur_shape = (1,cur_shape[1])
                cur_shape = (1, cur_shape[1])
                W = np.ascontiguousarray(W.astype(np.float32))
                b = np.ascontiguousarray(b.astype(np.float32))
                self.types.append('pool_conv')
                self.sizes.append(None)
                self.strides.append(stride)
                self.pads.append(pad)
                self.shapes.append(cur_shape)
                self.weights.append(W)
                self.biases.append(b)
            
            elif type(layer) == Activation or type(layer) == Lambda:
                print('activation')
                self.types.append('relu')
                self.sizes.append(None)
                self.strides.append(None)
                self.pads.append(None)
                self.shapes.append(cur_shape)
                self.weights.append(None)
                self.biases.append(None)
            elif type(layer) == InputLayer:
                print('input')
            elif type(layer) == BatchNormalization:
                print('batch normalization')
                gamma, beta, mean, std = weights
                std = np.sqrt(std+0.001) #Avoids zero division
                a = gamma/std
                b = -gamma*mean/std+beta
                
                print(np.shape(self.weights[-1]))
                self.weights[-1] = np.ascontiguousarray(a*self.weights[-1].transpose((1,2,0)).astype(np.float32))
                self.weights[-1] = np.ascontiguousarray(self.weights[-1].transpose((2,0,1)).astype(np.float32))
                self.biases[-1] = a*self.biases[-1]+b
            elif type(layer) == Dense:
                print('FC')
                W, b = weights
                b = b.astype(np.float32)
                W = W.reshape(list(cur_shape)+[W.shape[-1]]).astype(np.float32)
                cur_shape = (int((cur_shape[0]+pad[0]+pad[1]-W.shape[0])/stride[0])+1,W.shape[-1])
                W = np.ascontiguousarray(W.transpose((2,0,1)).astype(np.float32))
                b = np.ascontiguousarray(b.astype(np.float32))
                self.types.append('conv')
                self.sizes.append(None)
                self.strides.append((1,))
                self.pads.append((0,0,0,0))
                self.shapes.append(cur_shape)
                self.weights.append(W)
                self.biases.append(b)
            elif type(layer) == Dropout:
                print('dropout')
            
            elif type(layer) == GlobalMaxPooling1D:
                print('pool')
                pool_size = [num_points]
                stride = (1,)
                #padding = layer.get_config()['padding']
                pad = (0,0,0,0) #p_hl, p_hr, p_wl, p_wr
                if padding == 'same':
                    desired_h = int(np.ceil(cur_shape[0]/stride[0]))
                    desired_w = int(np.ceil(cur_shape[0]/stride[1]))
                    total_padding_h = stride[0]*(desired_h-1)+pool_size[0]-cur_shape[0]
                    total_padding_w = stride[1]*(desired_w-1)+pool_size[1]-cur_shape[1]
                    pad = (int(np.floor(total_padding_h/2)),int(np.ceil(total_padding_h/2)),int(np.floor(total_padding_w/2)),int(np.ceil(total_padding_w/2)))
                cur_shape = (1, cur_shape[1])
                self.types.append('pool')
                self.sizes.append(pool_size)
                self.strides.append(stride)
                self.pads.append(pad)
                self.shapes.append(cur_shape)
                self.weights.append(None)
                self.biases.append(None)
            elif type(layer) == Flatten:
                print('flatten')
            elif type(layer) == Reshape:
                print('reshape')
               
                cur_shape = (int(np.sqrt(cur_shape[1])),int(np.sqrt(cur_shape[1])))
                #cur_shape = (1,cur_shape[1])
                self.types.append('reshape')
                self.sizes.append(None)
                self.strides.append(None)
                self.pads.append(None)
                self.shapes.append(cur_shape)
                self.weights.append(None)
                self.biases.append(None)
            elif type(layer) == Dot:
                print('dot')
                #weights = model.get_layer(index=i-14).output
                cur_shape = self.shapes[0]
                b = np.zeros(cur_shape[0])
                b = np.ascontiguousarray(b.astype(np.float32))
                self.types.append('dot')
                self.sizes.append(None)
                self.strides.append((1,))
                self.pads.append((0,0,0,0))
                self.shapes.append(cur_shape)
                self.weights.append(None)
                self.biases.append(b)
            else:
                print('layer type is not defined', str(type(layer)))
                #raise ValueError('Invalid Layer Type')
        print(self.shapes)
        '''
        for i in range(2,len(self.weights)+1):
            print('Layer ' + str(i))
            print('types is',self.types[i])
            if self.weights[i] is not None:
                print(self.weights[i].shape)
        '''
    def predict(self, data):
        return self.model(data)


@njit
def UL_conv_bound(A, B, pad, stride, shape, W, b, inner_pad, inner_stride, inner_shape):
    #inner_shape = LBs.shape
    A_new = np.zeros((A.shape[0], A.shape[1], inner_stride[0]*(A.shape[2]-1)+W.shape[1], W.shape[2]), dtype=np.float32)
    B_new = B.copy()
    #print('conv A.shape',A.shape)
    #print('bias shape',b.shape)
    assert A.shape[3] == W.shape[0]
    for x in range(A_new.shape[0]):
        #p_start = np.maximum(0, 0-x)
        p_end = np.minimum(A.shape[2], shape[0]+x)
        
        t_end = np.minimum(A_new.shape[2], inner_shape[0]-inner_stride[0]*x)
        '''
        for y in range(A_new.shape[1]):
            q_start = np.maximum(0, pad[2]-stride[1]*y)
            q_end = np.minimum(A.shape[4], shape[1]+pad[2]-stride[1]*y)
            u_start = np.maximum(0, -stride[1]*inner_stride[1]*y+inner_stride[1]*pad[2]+inner_pad[2])
            u_end = np.minimum(A_new.shape[4], inner_shape[1]-stride[1]*inner_stride[1]*y+inner_stride[1]*pad[2]+inner_pad[2])
        '''
        for t in range(t_end):
            #for u in range(u_start, u_end):
                for p in range(0, p_end):
                    #for q in range(q_start, q_end):
                        if 0<=t-inner_stride[0]*p<W.shape[1] :
                            #print('conv A',A[x,:,p,:].shape)
                            #print('conv A',A[x,:,p,:].shape,file =f)
                            #print('W shape',W[:,t-inner_stride[0]*p,:].shape)
                            #print('W shape',W[:,t-inner_stride[0]*p,:].shape,file = f)
                            A_new[x,:,t,:] += np.dot(A[x,:,p,:],W[:,t-inner_stride[0]*p,:])
        for p in range(p_end):
            B_new[x,:] += np.dot(A[x,:,p,:],b)
    return A_new, B_new
@njit
def UL_pool_conv_bound(A, B, pad, stride, shape, W, b, inner_pad, inner_stride, inner_shape):
    #inner_shape = LBs.shape
    #A_new = np.zeros((A.shape[0], A.shape[1], W.shape[1], W.shape[2]), dtype=np.float32)
    A_new = np.zeros((A.shape[0], A.shape[1],W.shape[1]-A.shape[0]+1, W.shape[2]), dtype=np.float32)
    B_new = B.copy()
    
    #assert A.shape[3] == W.shape[0]
    #for t in range(A_new.shape[2]):
        #A_new [:,:,t,:]=A[:,:,0,:]*(1/W.shape[1])
    #A_new = A_new.repeat(64,axis = 2)
    
    for x in range(A_new.shape[0]):
        for y in range(A_new.shape[1]):
           # for i in range(A_new.shape[2]):
            for j in range(A.shape[3]):
                    A_new[x,y,:,j] = A[x,y,0,j]*(1/W.shape[1])
    
    
    return A_new, B_new



@njit
def UL_relu_bound(A, B, pad, stride, alpha_u, alpha_l, beta_u, beta_l):
    A_new = np.zeros_like(A)
    A_plus = np.maximum(A, 0)
    
    A_minus = np.minimum(A, 0)
    B_new = B.copy()
  
    for x in range(A_new.shape[0]):
        p_end = np.minimum(A.shape[2], alpha_u.shape[0]-x)
        for y in range(A_new.shape[1]):
           
            for p in range(p_end):
                for j in range(A.shape[3]):
                    if A[x,y,p,j] ==0:
                        pass
                    
                    else:
                 
                        A_new[x,y,p,j] +=A_minus[x,y,p,j]*alpha_l[p+x,j]+A_plus[x,y,p,j]*alpha_u[p+x,j]
                        B_new[x,y] += A_minus[x,y,p,j]*beta_l[p+x,j]+A_plus[x,y,p,j]*beta_u[p+x,j]
                    
                    
    return A_new, B_new

@njit
def UL_dot_bound(A_dot_u,B_dot_u,A_dot_l,B_dot_l,a,alpha,beta):
    A_new = np.zeros((1,alpha.shape[0], A_dot_u.shape[2],A_dot_u.shape[3]),dtype=np.float32) #A0(k,m,n) (a,k,m,n)
    beta_p = np.maximum(0,beta) #beta is x0-eps[a,:]
    beta_m = np.minimum(0,beta)
    
    # minus_minus = np.minimum(minus, 0)
    B_new = np.zeros((1,alpha.shape[0])) #alpha is (3,3)
    for j in range(alpha.shape[0]): #3
        for k in range(alpha.shape[0]):
            if beta[k] == 0:
                pass
            else:
                for m in range(A_new.shape[2]):
                    for n in range(A_new.shape[3]):
                        A_new[0,j,m,n]+=beta_p[k]*A_dot_u[0,3*k+j,m,n]+beta_m[k]*A_dot_l[0,3*k+j,m,n]
                A_new[0,j,a,k]+=alpha[k,j]
                B_new[0,j]+=beta_p[k]*B_dot_u[0,3*k+j]+beta_m[k]*B_dot_l[0,3*k+j]-beta[k]*alpha[k,j]

                           
    return A_new, B_new


 

@njit
def UL_dot_bound2(A,Au_new,Al_new,Bl_new,Bu_new):
    A_new = np.zeros((1,A.shape[0], Al_new.shape[2],Al_new.shape[3]),dtype=np.float32)
    A_p = np.maximum(0,A)
    A_m = np.minimum(0,A)
    B_new = np.zeros((A.shape[0]))
    
    for b in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[b,j]== 0:
                pass
            else:
                for m in range(A_new.shape[2]):
                    for n in range(A_new.shape[3]):
                        A_new[0,b,m,n] += A_p[b,j]*Au_new[0,j,m,n]+A_m[b,j]*Al_new[0,j,m,n]
                B_new[b]+=A_p[b,j]*Bu_new[0,j]+A_m[b,j]*Bl_new[0,j]
    return A_new,B_new



@njit
def UL_pool_bound(A, B, pad, stride, pool_size, inner_pad, inner_stride, inner_shape, alpha_u, alpha_l, beta_u, beta_l):
    A_new = np.zeros((A.shape[0], A.shape[1],  inner_stride[0]*(A.shape[2]-1)+pool_size[0],  A.shape[3]), dtype=np.float32)
    B_new = B.copy()
    A_plus = np.maximum(A, 0)
    A_minus = np.minimum(A, 0)

    for x in range(A_new.shape[0]):
            for t in range(A_new.shape[2]):
                    inner_index_x = t+stride*inner_stride[0]*x-inner_stride[0]*pad[0]-inner_pad[0]
                    if 0<=inner_index_x<inner_shape[0] :
                        for p in range(A.shape[2]):
                                if 0<=t-inner_stride[0]*p<alpha_u.shape[0] :
                                    A_new[x,:,t,:] += A_plus[x,:,p,:]*alpha_u[t-inner_stride[0]*p,p+stride*x-pad[0]]
                                    A_new[x,:,t,:] += A_minus[x,:,p,:]*alpha_l[t-inner_stride[0]*p,p+stride*x-pad[0]]
    B_new += conv_full(A_plus,beta_u,pad,stride) + conv_full(A_minus,beta_l,pad,stride)
    return A_new, B_new

@njit
def UL_joint_bound(A,B,A_add,B_add):
    A_new = np.zeros((A_add.shape[0],A_add.shape[1],A.shape[2]+A.shape[0]-1,A.shape[3]))
    B_new = B.copy()
    for a in range(A_new.shape[0]):
        for b in range(A_new.shape[1]):
            for i in range(A_add.shape[2]):
                for j in range(A_add.shape[3]):
                    for m in range(A.shape[0]):
                        for l in range(A.shape[2]):
                            for p in range(A.shape[3]):
                                if A_add[a,b,i,j] == 0:
                                    pass
                                elif A[m,j,l,p]==0:
                                    pass
                                else:
                                    A_new[a,b,m+l,p]+= A_add[a,b,i,j]*A[m,j,l,p]
                        B_new[a,b] +=A_add[a,b,i,j]*B[m,j] 
    return A_new, B_new

@njit
def UL_add_bound(A,B,A_m,B_m,A_add,B_add):
    A_new = np.zeros((A.shape[2]+A.shape[0]-1,A.shape[3]))
    B_new = B_add
    A_add_p = np.maximum(0,A_add)
    A_add_m = np.minimum(0,A_add)
    for i in range(A_add.shape[0]):
        for j in range(A_add.shape[1]):
            for m in range(A.shape[0]):
                for l in range(A.shape[2]):
                    for p in range(A.shape[3]):
                        if A_add[i,j] == 0:
                            pass
                        elif A[m,j,l,p]==0:
                            pass
                        else:
                            A_new[m+l,p]+= A_add_p[i,j]*A[m,j,l,p] + A_add_m[i,j]*A_m[m,j,l,p]
                            #A_new[m+l,p]+= A_add[i,j]*A[m,j,l,p]
                B_new +=A_add_p[i,j]*B[m,j] + A_add_m[i,j]*B_m[m,j] 
                #B_new +=A_add[i,j]*B[m,j]
    return A_new, B_new

#Main function to find bounds at each layer

def compute_bounds(weights, biases, out_shape, nlayer, x0, eps, p_n, pads, strides, sizes, types, LBs, UBs,last_Au,last_Al,last_Bu,last_bl,dot_A_u=None, dot_A_l=None,dot_B_u=None,dot_B_l=None ):

    
    repeat = False
    bia = False
    pool = False
    beta = False
    add = False
    dot = False
    if types[nlayer-1] == 'relu':
        #print('relu')
        #print('relu result is',np.maximum(LBs[nlayer-1], 0)[0][0])
        return np.maximum(LBs[nlayer-1], 0), np.maximum(UBs[nlayer-1], 0),last_Au,last_Al,last_Bu,last_bl,repeat
    elif types[nlayer-1] == 'reshape':
        #print('reshape')
        #print('reshape LB is',LBs[nlayer-1].shape)
        repeat = True
        
        print('last au',last_Au.shape)
        return np.reshape(LBs[nlayer-1],(3,3)) ,np.reshape(UBs[nlayer-1],(3,3)),last_Au,last_Al,last_Bu,last_bl,repeat

        #return np.tile(LBs[nlayer-1],num_points).reshape(num_points,-1),np.tile(UBs[nlayer-1],num_points).reshape(num_points,-1)
    
    elif types[nlayer-1] == 'dot':
        #print('dot')
        lx,ux = LBs[0],UBs[0]
        
        #ly,uy = np.reshape(LBs[nlayer-1][0],(3,3)), np.reshape(UBs[nlayer-1][0],(3,3))
        ly, uy = LBs[nlayer-1],UBs[nlayer-1]
        
       
        #lg= -lx*ly
        #ug = -lx*uy
        LB = np.zeros_like(LBs[0],dtype=np.float32)
        UB = np.zeros_like(UBs[0],dtype=np.float32)
        #print('lx.shape',lx.shape)
        #print('ly.shape',ly.shape)
        #print('LB.shape',LB.shape)
        
        #LB = np.dot(lx,LBs[nlayer-1])
        #UB = np.dot(ux,UBs[nlayer-1])
        
        for i in range(lx.shape[0]):
            for j in range(lx.shape[1]):
                for k in range(lx.shape[1]):
        
                    if ux[i,k] <0:
                        LB[i,j] += lx[i,k]*uy[k,j]
                        UB[i,j] += ux[i,k]*ly[k,j]
                    elif lx[i,k]>0:
                        UB[i,j] += ux[i,k]*uy[k,j]
                        LB[i,j] += lx[i,k]*ly[k,j]
                    else:
                        UB[i,j] += ux[i,k]*uy[k,j]
                        LB[i,j] += lx[i,k]*ly[k,j]
        
        pad = (0,0,0,0)
        stride = 1
        #print(LB[0][0])
        return LB,UB,last_Au,last_Al,last_Bu,last_bl,repeat
        
    elif types[nlayer-1] == 'conv':
        #print('conv',file =f)
        
        #print('conv')
        
        
        A_u = weights[nlayer-1].reshape((1, weights[nlayer-1].shape[0], weights[nlayer-1].shape[1], weights[nlayer-1].shape[2]))*np.ones((out_shape[0], weights[nlayer-1].shape[0], weights[nlayer-1].shape[1], weights[nlayer-1].shape[2]), dtype=np.float32)
        #print('A_u.shape',A_u.shape)
        
        #print('outshape',out_shape)
        B_u = biases[nlayer-1]*np.ones((out_shape[0], out_shape[1]), dtype=np.float32)
        
        A_l = A_u.copy()
        B_l = B_u.copy()
        pad = pads[nlayer-1]
        stride = 1
        #print('output value for conv is',conv_bound_full(A_u, B_u, pad, stride,UBs[nlayer-1] , 0.000, 1)[0][0])


    elif types[nlayer-1] ==  'pool_conv':
        #print('pool_conv',file =f )
        #print('pool_conv')
        #print(UBs[nlayer-2])
        #print(UBs[nlayer-1])
        #print('average value is',np.average(UBs[nlayer-1],axis = 0))
        A_u = weights[nlayer-1].reshape((1, weights[nlayer-1].shape[0], weights[nlayer-1].shape[1], weights[nlayer-1].shape[2]))*np.ones((out_shape[0], weights[nlayer-1].shape[0], x0.shape[0]-out_shape[0]+1, weights[nlayer-1].shape[2]), dtype=np.float32)
        #print('weights.shape',weights[nlayer-1].shape)
        #print('A_u.shape',A_u.shape)
        #print('outshape',out_shape)
        
        # A_u = np.repeat(A_u,num_points,axis = 2)
        B_u = biases[nlayer-1]*np.ones((out_shape[0], out_shape[1]), dtype=np.float32)
        
        A_l = A_u.copy()
        B_l = B_u.copy()
        pad = pads[nlayer-1]
        stride = 1
        #print('output value for pool_conv is',conv_bound_full(A_u, B_u, pad, stride,UBs[nlayer-1] , 0.000, 1)[0][0])
    elif types[nlayer-1] == 'pool':
        #print('pool')
        #print('pool',file =f)
        A_u = np.eye(out_shape[1]).astype(np.float32).reshape((1,out_shape[1],1,out_shape[1]))*np.ones((out_shape[0], out_shape[1], 1,out_shape[1]), dtype=np.float32)
        B_u = np.zeros(out_shape, dtype=np.float32)
        A_l = A_u.copy()
        B_l = B_u.copy()
        pad = (0,0,0,0)
        stride = 1
        #pool_size = weights[nlayer-1].shape[1:]
        pool_size = [num_points]
        alpha_u, alpha_l, beta_u, beta_l = pool_linear_bounds(LBs[nlayer-1], UBs[nlayer-1], pads[nlayer-1], np.asarray(strides[nlayer-1]),  pool_size)
        
        A_u, B_u = UL_pool_bound(A_u, B_u, np.asarray(pad), np.asarray(stride), pool_size, np.asarray(pads[nlayer-1]), np.asarray(strides[nlayer-1]), np.asarray(LBs[nlayer-1].shape), alpha_u, alpha_l, beta_u, beta_l)
        A_l, B_l = UL_pool_bound(A_l, B_l, np.asarray(pad), np.asarray(stride), pool_size, np.asarray(pads[nlayer-1]), np.asarray(strides[nlayer-1]), np.asarray(LBs[nlayer-1].shape), alpha_l, alpha_u, beta_l, beta_u)
    

    
    for i in range(nlayer-2, -1, -1):

        if types[i] == 'conv':
   
    
            
            
            #print('output value for conv is',conv_bound_full(A_u, B_u, pad, stride,UBs[i+1] , 0.000, 1))
            A_u, B_u = UL_conv_bound(A_u, B_u, np.asarray(pad), np.asarray(stride), np.asarray(UBs[i+1].shape), weights[i], biases[i], np.asarray(pads[i]), np.asarray(strides[i]), np.asarray(UBs[i].shape))
            #print('A_u',A_u.shape)
            A_l, B_l = UL_conv_bound(A_l, B_l, np.asarray(pad), np.asarray(stride), np.asarray(LBs[i+1].shape), weights[i], biases[i], np.asarray(pads[i]), np.asarray(strides[i]), np.asarray(LBs[i].shape))
            #print('output value for conv is',conv_bound_full(A_u, B_u, pad, stride,UBs[i+1] , 0.000, 1))
            
            #if pool == True:
                #print('output value for after conv is',conv_bound_full(A_u, B_u, pad, stride,UBs[i] , 0.000, 1,pool,bia)[0][0][0])
            #print('A_u.shape',A_u.shape,file = f)
            #print('A_l.shape',A_l.shape,file = f)
            pad = (0,0,0,0)
            stride = 1
        if types[i] == 'pool_conv':
            #print('pool_conv')
            #print('pool_conv',file = f)
            #print('weights.shape',weights[i].shape)
            #print('UBs[i].shape',UBs[i].shape)
            #print('prev A_u',A_u.shape)
            #if A_u.shape[3] != weights[i].shape[0]:
                #print('UBS i-14',UBs[i-14].shape)
            repeat = False   
            #print('dot for this layer is',dot)
            
            
            #print('output value for pool is',conv_bound_full(A_u, B_u, pad, stride,UBs[i+1] , 0.000, 1)[0][0][0])
            A_u, B_u = UL_pool_conv_bound(A_u, B_u, np.asarray(pad), np.asarray(stride), np.asarray(UBs[i+1].shape), weights[i], biases[i], np.asarray(pads[i]), np.asarray(strides[i]), np.asarray(UBs[i].shape))
            A_l, B_l = UL_pool_conv_bound(A_l, B_l, np.asarray(pad), np.asarray(stride), np.asarray(LBs[i+1].shape), weights[i], biases[i], np.asarray(pads[i]), np.asarray(strides[i]), np.asarray(LBs[i].shape))
            #print('output value for pool is',conv_bound_full(A_u, B_u, pad, stride,UBs[i] , 0.000, 1)[0][0][0])
            
            #if dot == True:
                #print('output value for pool is', conv_bound_full(A_u, B_u,pad,stride,UBs[i],0.000, 1,pool)[0][0][0])
            
            pad = (0,0,0,0)
            stride = 1
        elif types[i] == 'pool':
            #print('pool')
            #pool_size = weights[i].shape[-1:]
            pool_size = [num_points]
            #print('prev A_u',A_u.shape)
            alpha_u, alpha_l, beta_u, beta_l = pool_linear_bounds(LBs[i], UBs[i], np.asarray(pads[i]), np.asarray(strides[i]),  pool_size)
            
            
            A_u, B_u = UL_pool_bound(A_u, B_u, np.asarray(pad), np.asarray(stride), pool_size, np.asarray(pads[i]), np.asarray(strides[i]), np.asarray(UBs[i].shape), alpha_u, alpha_l, beta_u, beta_l)
            #print('A_u',A_u.shape)
            A_l, B_l = UL_pool_bound(A_l, B_l, np.asarray(pad), np.asarray(stride), pool_size, np.asarray(pads[i]), np.asarray(strides[i]), np.asarray(LBs[i].shape), alpha_l, alpha_u, beta_l, beta_u)
            
            pad = (0,0,0,0)
            stride = 1
        
        elif types[i] == 'relu':
            #print('relu')
            #print('relu',file =f)
            
            #print('start Relu bound') 
  
            
            alpha_u, alpha_l, beta_u, beta_l = linear_bounds(LBs[i], UBs[i])
            A_u, B_u = UL_relu_bound(A_u, B_u, np.asarray(pad), np.asarray(stride), alpha_u, alpha_l, beta_u, beta_l)
            #if dot == True and pool ==False:
                #print('output value after relu is',conv_bound_full(A_u, B_u, pad, stride,np.tile(UBs[i],64).reshape(64,-1) , 0.000, 1))
            start_time = time.time()
            #print(A_u,file =f)
            #print('end Relu bound')
            A_l, B_l = UL_relu_bound(A_l, B_l, np.asarray(pad), np.asarray(stride), alpha_l, alpha_u, beta_l, beta_u)
           
            #print('total time',time.time()-start_time)
            #print('total time',time.time()-start_time,file = f)
            #print('end Relu bound')
            #print('A_u.shape',A_u.shape)
            #print('A_l,shape',A_l.shape,file =f )
            
            
 
            
            
        elif types[i] == 'dot':
            print('dot')
            
            dot = True
        
            
            UUB = np.zeros((A_u.shape[0],A_u.shape[1]))
            LLB = UUB.copy()
            start_t = time.time()
            for a in range(A_u.shape[0]):
                A_new_U = np.zeros((1,A_u.shape[1], x0.shape[0],x0.shape[1]),dtype=np.float32)
                B_new_U = B_u[a,:].copy()
                A_new_l = A_new_U.copy()
                B_new_l = B_l[a,:].copy()
                
                
                
                for h in range(A_u.shape[2]):
                    Au_new, Bu_new = UL_dot_bound(dot_A_u,dot_B_u,dot_A_l,dot_B_l,a+h,UBs[i],x0[a+h]) #LBs[0][a+h]
                    Al_new, Bl_new= UL_dot_bound(dot_A_l,dot_B_l,dot_A_u,dot_B_u,a+h,LBs[i],x0[a+h]) #LBs[0][a+h]
                    #print('upper is', conv_bound_full(A_new_U,np.expand_dims(B_new_U,0), pad, stride, x0[a+h], eps,p_n))
                    #print('lower is',conv_bound_full(A_new_l,np.expand_dims(B_new_l,0), pad, stride, x0[a+h], eps, p_n))

                    A_1,B_1 = UL_dot_bound2(A_u[a,:,h,:],Au_new,Al_new,Bu_new,Bl_new)
                    A_new_U += A_1
                    B_new_U += B_1
                    A_2,B_2 = UL_dot_bound2(A_l[a,:,h,:],Al_new,Au_new,Bl_new,Bu_new)
                    A_new_l += A_2
                    B_new_l += B_2
                
                
                
                _, UUB[a,:] = conv_bound_full(A_new_U,np.expand_dims(B_new_U,0), pad, stride, x0, eps,p_n)
                LLB[a,:], _ = conv_bound_full(A_new_l,np.expand_dims(B_new_l,0), pad, stride, x0, eps, p_n)
            #print('time is',time.time()-start_t)
            return LLB, UUB, A_u, A_l,B_u,B_l,repeat

   
    LUB, UUB = conv_bound_full(A_u, B_u, pad, stride, x0, eps, p_n)
        #LLB, ULB = conv_bound_full(A_l, B_l, pad, stride, x0, eps, p_n,np.zeros_like(A_l),np.zeros_like(A_u),dot)
        #print('LLB',np.max(LLB[-1]),file = f)
    LLB, ULB = conv_bound_full(A_l, B_l, pad, stride, x0, eps, p_n)
        #if dot == True:
    #print('add is',add)    
    
    #print('layer result is',UUB[0][0])
    return LLB, UUB, A_u, A_l,B_u,B_l,repeat

#Main function to find output bounds
def find_output_bounds(weights, biases, shapes, pads, strides, sizes, types, x0, eps, p_n):
    #LB, UB = conv_bound(weights[0], biases[0], pads[0], strides[0], x0, eps, p_n)
    #for i in range(len(x0)):
        #x0[i] = x0
    LBs = [x0-eps]
    UBs = [x0+eps]
    dots = False
    A_u, A_l,B_u,B_l = None,None,None, None
    dot_A_u,dot_A_l, dot_B_u,dot_B_l = None,None,None, None
    
    for i in range(1,len(weights)+1):
        #print('Layer ' + str(i))
        #print('types i',types[i-1],file = f)
        repeat = False
        #st = time.time()
        LB, UB,A_u, A_l,B_u,B_l,repeat = compute_bounds(weights, biases, shapes[i], i, x0, eps, p_n, pads, strides, sizes, types, LBs, UBs,A_u, A_l,B_u,B_l,dot_A_u,dot_A_l, dot_B_u,dot_B_l)
        #print('total time',time.time()-st)
        if repeat == True:
            dot_A_u,dot_A_l, dot_B_u,dot_B_l = A_u, A_l,B_u,B_l
        UBs.append(UB)
        LBs.append(LB)
        #print('UBs[-1]',np.max(UBs[-1]),file = f)
        #print('LBs[-1]',np.max(LBs[-1]),file = f)
    return LBs[-1], UBs[-1]
def verifyMaximumEps(classifier, x0, eps, p,true_label, target_label,
                        eps_idx = None, untargeted=False, thred=1e-4):
    
    
    y0 = classifier(np.expand_dims(x0,axis=0))
    out0 = y0[true_label]
    max_iter = 100
    fail =  0
    for i in range(max_iter):
        noise = generateNoise(x0.shape, p, eps)
        y = classifier(np.expand_dims(x0+noise,axis=0))[0]
    
        if not untargeted:
            out = y[target_label]
        else:
            y[true_label] = y[true_label] - 1e8
            out = np.max(y)
            
        valid = (out0 + thred >= out)
        print('iter %d true-target min %.4f' % (i, (out0-out).min()))
        if valid.min() < 1:
            print('failed')
            fail += 1
    return fail



#Warms up numba functions
def warmup(model, x, eps_0, p_n, func):
    print('Warming up...')
    weights = model.weights[:-1]
    
    biases = model.biases[:-1]
    shapes = model.shapes[:-1]
    print(shapes)
    W, b, s = model.weights[-1], model.biases[-1], model.shapes[-1]
    last_weight = np.ascontiguousarray((W[0,:,:]).reshape([1]+list(W.shape[1:])),dtype=np.float32)
    weights.append(last_weight)
    biases.append(np.asarray([b[0]]))
    shapes.append((1,1))
    func(weights, biases, shapes, model.pads, model.strides, model.sizes, model.types, x, eps_0, p_n)

#Main function to compute CNN-Cert bound
def run(model, inputs, targets, true_labels, true_ids, img_info,n_samples, p_n, q_n, activation = 'relu', cifar=False, tinyimagenet=False):
    np.random.seed(10)
    f = open('./very-max-15layer'+str(p_n)+'.txt', "a+")
    #print('inputs.shape',inputs.shape,file=f)
    tf.set_random_seed(10)
    random.seed(10)
    #keras_model = load_model(file_name, custom_objects={'fn':loss, 'ResidualStart':ResidualStart, 'ResidualStart2':ResidualStart2, 'tf':tf})
    keras_model = model
    if tinyimagenet:
        model = Model(keras_model, inp_shape = (64,64,3))
    elif cifar:
        model = Model(keras_model, inp_shape = (32,32,3))
    else:
        model = Model(keras_model)

    #Set correct linear_bounds function
    global linear_bounds
    if activation == 'relu':
        linear_bounds = relu_linear_bounds
    elif activation == 'ada':
        linear_bounds = ada_linear_bounds
    elif activation == 'sigmoid':
        linear_bounds = sigmoid_linear_bounds
    elif activation == 'tanh':
        linear_bounds = tanh_linear_bounds
    elif activation == 'arctan':
        linear_bounds = atan_linear_bounds
    
    

    if len(inputs) == 0:
        return 0, 0
    
    #0b01111 <- all
    #0b0010 <- random
    #0b0001 <- top2
    #0b0100 <- least
    preds = model.model.predict(inputs[0][np.newaxis,:]).flatten()
    steps = 10
    eps_0 = 0.05
    summation = 0

    #warmup(model, inputs[0].astype(np.float32), eps_0, p_n, find_output_bounds)
    inp = model.model.input                                    # input placeholder
    outputs = [layer.output for layer in model.model.layers] 
    outputs = outputs[1:]         # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs]    # evaluation functions  
    start_time = time.time()
    L= len(inputs)
    for i in range(len(inputs)):
        
        #layer_outs = [func([np.expand_dims(inputs[i],axis = 0)]) for func in functors]
        
        #print('layer output shape is',len(layer_outs))
        #for a in range(len(layer_outs)):
            
            #print(layer_outs[a][0][0],file =f)

        print('--- CNN-Cert: Computing eps for input image ' + str(i)+ '---')
        predict_label = true_labels[i]
        #print('predictlabel',  predict_label,file =f)
        target_label = np.argmax(targets[i])
        #print('target label',target_label,file =f)
        predict_prob = np.squeeze(model.model.predict(np.expand_dims(inputs[i],axis = 0)))
        #print('predict_prob',predict_prob[target_label])
        ''' 
        if predict_prob[predict_label]-predict_prob[target_label] < 1:
            print('prob loss',predict_prob[predict_label]-predict_prob[target_label],file = f)
            print('prob loss',predict_prob[predict_label]-predict_prob[target_label])
        else:
            continue
        '''
        print('###########################################',file =f)
        print( 'predict_label', predict_prob[predict_label],file =f)
        print( 'target_label', predict_prob[target_label],file =f)
        print('predictlabel',  predict_label,file =f)
        print('target label',target_label,file =f)
        weights = model.weights[:-1]
        biases = model.biases[:-1]
        shapes = model.shapes[:-1]
        W, b, s = model.weights[-1], model.biases[-1], model.shapes[-1]
        print('W shape',W.shape,file =f)
        last_weight = (W[predict_label,:,:]-W[target_label,:,:]).reshape([1]+list(W.shape[1:]))
        
        #last_weight = (W[predict_label,:,:]).reshape([1]+list(W.shape[1:]))
        weights.append(last_weight)
        biases.append(np.asarray([b[predict_label]-b[target_label]]))
        shapes.append((1,1))

        #Perform binary searchcc
        log_eps = np.log(eps_0)
        log_eps_min = -np.inf
        log_eps_max = np.inf
        faild = False
        print( 'predict_label',model.model.predict(np.expand_dims(inputs[i],axis = 0))[0][predict_label])
        print( 'target_label',model.model.predict(np.expand_dims(inputs[i],axis = 0))[0][target_label])
        for j in range(steps):
            #print('Step ' + str(j))
            sttime=time.time()
            LB, UB = find_output_bounds(weights, biases, shapes, model.pads, model.strides, model.sizes, model.types, inputs[i].astype(np.float32), np.exp(log_eps), p_n)
            roud_time=time.time()-sttime
            print("Step {}, eps = {:.5f}, {:.6s} <= f_c - f_t <= {:.6s},time = {:.5f}".format(j,np.exp(log_eps),str(np.squeeze(LB)),str(np.squeeze(UB)),roud_time))
            print("Step {}, eps = {:.5f}, {:.6s} <= f_c - f_t <= {:.6s},time = {:.5f}".format(j,np.exp(log_eps),str(np.squeeze(LB)),str(np.squeeze(UB)),roud_time),file=f)
            if LB > 0: #Increase eps
                log_eps_min = log_eps
                log_eps = np.minimum(log_eps+1, (log_eps_max+log_eps_min)/2)
            
            else: #Decrease eps
                log_eps_max = log_eps
                log_eps = np.maximum(log_eps-1, (log_eps_max+log_eps_min)/2)
            if np.exp(log_eps) >10:
                faild = True
                L=L-1
                break
        if p_n == 105:
            str_p_n = 'i'
        else:
            str_p_n = str(p_n)
        print('prob loss',predict_prob[predict_label]-predict_prob[target_label])
        print('prob loss',predict_prob[predict_label]-predict_prob[target_label],file = f)
        print("[L1] method = CNN-Cert-{},  image no = {}, true_id = {}, target_label = {}, true_label = {}, norm = {}, robustness = {:.5f}".format(activation, i, true_ids[i],target_label,predict_label,str_p_n,np.exp(log_eps_min)))
        print("[L1] method = CNN-Cert-{},  image no = {}, true_id = {}, target_label = {}, true_label = {}, norm = {}, robustness = {:.5f}".format(activation, i, true_ids[i],target_label,predict_label,str_p_n,np.exp(log_eps_min)),file=f)
       
        if faild == False:
            summation += np.exp(log_eps_min)
    K.clear_session()
    
    eps_avg = summation/L
    total_time = (time.time()-start_time)/len(inputs)
    print('L is',L)
    print("[L0] method = CNN-Cert-{},  total images = {}, norm = {}, avg robustness = {:.5f}, avg runtime = {:.2f}".format(activation,len(inputs),str_p_n,eps_avg,total_time))
    print("[L0] method = CNN-Cert-{},  total images = {}, norm = {}, avg robustness = {:.5f}, avg runtime = {:.2f}".format(activation,len(inputs),str_p_n,eps_avg,total_time),file=f)
    return eps_avg, total_time
    
