# -*- coding: utf-8 -*-
"""
Created on Wed May 23 14:15:46 2018

@author: yogurts
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib.pyplot as plt
import scipy
import math
'''
Version 2.0
Add   L2 regularization, Drop out
use Adam and dynamic learning rate and mini batch to speed up.
'''
def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s


def relu(x):
    """
    Compute the relu of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- relu(x)
    """
    s = np.maximum(0,x)

    return s



def initialization_parameter_random(layer_dim):
    '''
    initialize parameters
    input: layer dimensional(list)
    output: parameters, which contain the parameter for your neural network
    W1,b1--W2,b2--W3,b3--------WL,bL
    W: weight matrix of shape
    b: bias vector of shape
    Note: here I use "He initialization" He et al 2015 to do initialization, multiply the W with a variable, which depends on the number of last layer
    '''
    parameters = {}
    L = len(layer_dim)  # number of the layer
    
    for i in range(1, L):
        parameters['W'+str(i)] = np.random.randn(layer_dim[i], layer_dim[i-1])*np.sqrt(2/layer_dim[i-1])
        parameters['b'+str(i)] = np.zeros((layer_dim[i], 1))
    return parameters
 
def random_mini_bacth(X, Y, mini_batch_size = 64):
   '''
   use mini bacth to speed up
   '''
   m = X.shape[1]
   mini_batches = []
   '''
   step 1: shuffle input X and labels Y
   '''
   permutation = list(np.random.permutation(m))
   shuffled_X = X[:,permutation]
   shuffled_Y = Y[:,permutation].reshape((1,m))
   num_mini = math.floor(m/mini_batch_size)
   '''
   partion
   '''
   for k in range(0, num_mini):
       mini_batch_X = shuffled_X[:,k*mini_batch_size:(k+1)*mini_batch_size]
       mini_batch_Y = shuffled_Y[:,k*mini_batch_size:(k+1)*mini_batch_size]
       mini_batch = (mini_batch_X, mini_batch_Y)
       mini_batches.append(mini_batch)
   '''
   save the last part
   '''
   if m% mini_batch_size != 0:
      mini_batch_X = shuffled_X[:,num_mini*mini_batch_size:m]
      mini_batch_Y = shuffled_Y[:,num_mini*mini_batch_size:m]
      mini_batch = (mini_batch_X, mini_batch_Y)
      mini_batches.append(mini_batch)
      
   return mini_batches

def initialize_Adam(parameters):
   L = len(parameters)//2
   v = {}
   s = {}      
   for i in range(L):
      v['dW'+str(i+1)] = np.zeros(parameters["W"+str(i+1)].shape)
      v['db'+str(i+1)] = np.zeros(parameters["b"+str(i+1)].shape)
      s['dW'+str(i+1)] = np.zeros(parameters["W"+str(i+1)].shape)
      s['db'+str(i+1)] = np.zeros(parameters["b"+str(i+1)].shape)
   return v,s

def forward_propagation(X, parameters):
    """
    Implements the forward propagation (and computes the loss) presented in Figure 2.

    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape ()
                    b1 -- bias vector of shape ()
                    W2 -- weight matrix of shape ()
                    b2 -- bias vector of shape ()
                    W3 -- weight matrix of shape ()
                    b3 -- bias vector of shape ()

    Returns:
    loss -- the loss function (vanilla logistic loss)
    """

    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    W4 = parameters["W4"]
    b4 = parameters["b4"]
    

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    z1 = np.dot(W1, X) + b1
    a1 = relu(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = relu(z2)
    z3 = np.dot(W3, a2) + b3
    a3 = relu(z3)
    
    z4 = np.dot(W4, a3) + b4
    AL = sigmoid(z4)
    cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3, z4, AL, W4, b4)

    return AL, cache
 
def compute_loss(AL, Y):

    """
    Implement the loss function

    Arguments:
    a3 -- post-activation, output of forward propagation
    Y -- "true" labels vector, same shape as a3

    Returns:
    loss - value of the loss function
    """

    m = Y.shape[1]
    logprobs = np.multiply(-np.log(AL),Y) + np.multiply(-np.log(1 - AL), 1 - Y)
    loss = 1./m * np.nansum(logprobs)

    return loss
 
def compute_cost_L2(AL, Y, parameters, lambd):
   m = Y.shape[1]
   L = len(parameters)//2
   cross_entropy_cost = compute_loss(AL, Y)
   L2_cost = 0
   for i in range(L):
      W_Frobenius = np.sum(np.square(parameters['W'+str(i+1)]))
      L2_cost += W_Frobenius
   cost = cross_entropy_cost+L2_cost*(1./m*lambd/2)
   return cost
 
def backward_propagation(X, Y, cache):
    """
    Implement the backward propagation presented in figure 2.

    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
    cache -- cache output from forward_propagation()

    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    m = X.shape[1]
    (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3, z4, AL, W4, b4) = cache

    dz4 = 1./m * (AL - Y)
    dW4 = np.dot(dz4, a3.T)
    db4 = np.sum(dz4, axis=1, keepdims = True)
    
    da3 = np.dot(W4.T, dz4)
    dz3 = np.multiply(da3, np.int64(a3 > 0))
    dW3 = np.dot(dz3, a2.T)
    db3 = np.sum(dz3, axis=1, keepdims = True)

    da2 = np.dot(W3.T, dz3)
    dz2 = np.multiply(da2, np.int64(a2 > 0))
    dW2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims = True)

    da1 = np.dot(W2.T, dz2)
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dW1 = np.dot(dz1, X.T)
    db1 = np.sum(dz1, axis=1, keepdims = True)

    gradients = {"dz4": dz4, "dW4": dW4, "db4": db4,
                 "da3": da3, "dz3": dz3, "dW3": dW3, "db3": db3,
                 "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
                 "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}

    return gradients

def predict(X, y, parameters):
    """
    This function is used to predict the results of a  n-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    p = np.zeros((1,m), dtype = np.int)

    # Forward propagation
    a3, caches = forward_propagation(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, a3.shape[1]):
        if a3[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    # print results

    #print ("predictions: " + str(p[0,:]))
    #print ("true labels: " + str(y[0,:]))
    print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))

    return p

def update_parameters_Adam(parameters, grads, v, s, t, learning_rate = 0.01,beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
   
   '''
   update parameters with Adam algorithm
   '''
   
   L = len(parameters)//2
   v_corrected = {}
   s_corrected = {}
   
   for i in range(L):
      
      '''
      compute 
      '''
      v['dW'+str(i+1)] = beta1 * v['dW'+str(i+1)]+(1-beta1) * grads['dW'+str(i+1)]
      v['db'+str(i+1)] = beta1 * v['db'+str(i+1)]+(1-beta1) * grads['db'+str(i+1)]
      v_corrected['dW'+str(i+1)] = v['dW'+str(i+1)] / (1-beta1**t)
      v_corrected['db'+str(i+1)] = v['db'+str(i+1)] / (1-beta1**t)
      s['dW'+str(i+1)] = beta2 * s['dW'+str(i+1)]+(1-beta2) * (grads['dW'+str(i+1)]**2)
      s['db'+str(i+1)] = beta2 * s['db'+str(i+1)]+(1-beta2) * (grads['db'+str(i+1)]**2)
      s_corrected['dW'+str(i+1)] = s['dW'+str(i+1)] / (1-beta2**t)
      s_corrected['db'+str(i+1)] = s['db'+str(i+1)] / (1-beta2**t)
      '''
      update parameters
      '''
      parameters['W'+str(i+1)]=parameters['W'+str(i+1)]-learning_rate*(v_corrected['dW'+str(i+1)]/(np.sqrt(s_corrected['dW'+str(i+1)])+epsilon))
      parameters['b'+str(i+1)]=parameters['b'+str(i+1)]-learning_rate*(v_corrected['db'+str(i+1)]/(np.sqrt(s_corrected['db'+str(i+1)])+epsilon))
      
   return parameters,v,s

def model(X, Y, layers_dims, learning_rate = 1, mini_batch_size = 64,
          beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, num_epochs = 1000, decay_rate =0.01, print_cost = True):

   L = len(layers_dims)
   costs = []
   t = 0
   parameters = initialization_parameter_random(layers_dims)
   v,s = initialize_Adam(parameters)
   
   for i in range(num_epochs):
      minibatches = random_mini_bacth(X, Y, mini_batch_size)
      learning_rate *= 1/(1+decay_rate*i) 
      for minibatch in minibatches:
         (minibatch_X, minibatch_Y) = minibatch
         AL,caches = forward_propagation(minibatch_X, parameters)
         cost = compute_loss(AL, minibatch_Y)
         #compute_cost_L2(AL, Y, parameters, lambd)
         grads = backward_propagation(minibatch_X, minibatch_Y, caches)
         #L_model_backward_L2(AL, Y, caches,parameters, lambd)
         t += 1
         parameters,v,s = update_parameters_Adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)
      if print_cost and i % 2 == 0:
         print ("Cost after epoch %i: %f" %(i, cost))
      if print_cost and i % 2 == 0:
         costs.append(cost)   
   plt.plot(costs)
   plt.ylabel('cost')
   plt.xlabel('epochs (per 100)')
   plt.title("Learning rate = " + str(learning_rate))
   plt.show()
   return parameters

if __name__ == "__main__":
    file=h5py.File('D:\\sa\\dataforversion2\\TrainSet_1001.h5','r')
    X = file['EEG_1'][:]
    Y = file['train_set_num'][:]
    file.close()
    layers_dims = [3000,16,8,4,1]  #5 layers
   # X = EEG_raw[:,0:10000]
    #Y = Labels_raw[0:10000,:]
    #X_test = EEG_raw[:,10000:]
    #Y_test = Labels_raw[10000:,:]
    #del EEG_raw, Labels_raw
    parameters =  model(X, Y, layers_dims, learning_rate = 1, mini_batch_size = 64, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, num_epochs = 1000, decay_rate =0.01, print_cost = True)
    p = predict(X, Y, parameters)
    file = h5py.File('Parameters.h5' ,'w')
    # 写入
    file.create_dataset('parameters', data = parameters)
 
         
         
         
      
      
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   