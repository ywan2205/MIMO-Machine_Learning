# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 17:36:16 2019

@author: czho6957
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import scipy.io as spio

#=============================================================
# load train and test data, label  
#=============================================================

file1 = spio.loadmat('train_and_test_data1.mat')
train_data_real = file1['y6_save_real_image'][:]
train_data_real = np.transpose(train_data_real)

train_data_real = np.array(train_data_real).astype(np.float32)

file3 = spio.loadmat('label1.mat')
print(file3)
Label = file3['parameter_tt'][:]

Label = np.array(Label).astype(np.float32)

print(Label)
train_data1 = train_data_real[:15000, :, :]
test_data1 = train_data_real[15000:, :, :]

train_label1 = Label[1,:15000]
test_label1 = Label[1,15000:]
    
# setting Parameters
learning_rate = 0.001
num_steps = 5000
batch_size = 10

# Network Parameters
n_input = 32*20
n_classes = 1 
dropout = 0.75 # Dropout, probability to keep units

# Create the neural network
def conv_net(x_dict, n_classes, dropout, reuse, is_training):#what is reuse，reuse is that it will continue use former number on last layer
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse = reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['train_and_test_data1']
        
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 32, 20, 1])
        #张数，维度和深度

        # Convolution Layer1 with 32 filters and a kernel size of 2
        conv1 = tf.layers.conv2d(
        inputs = x,
        filters = 32,
        kernel_size = [3, 3],
        strides=(1, 1),
        padding = 'same',
        activation = tf.nn.relu )
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        pool1 = tf.layers.max_pooling2d(conv1, 2, 2)
        #[-1,16,10,32]
        # Convolution Layer2 with 64 filters and a kernel size of 2
        conv2 = tf.layers.conv2d(
        inputs = pool1, 
        filters = 64, 
        kernel_size = [3, 3], 
        strides=(1, 1),
        padding = 'same',
        activation = tf.nn.relu )
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
        #[-1,8,5,64]
        # ===== new layer1
        #Convolution Layer3 with 64 filters and a kernel size of 2 
        conv3 = tf.layers.conv2d(
        inputs = pool2, 
        filters = 128, 
        kernel_size = [3, 3], 
        strides=(1, 1),
        padding = 'same',
        activation = tf.nn.relu )
        # Max Pooling (down-sampling) with strides of 1 and kernel size of 1
        pool3 = tf.layers.max_pooling2d(conv3, 2, 1)
        #[-1,4,5,128]
         
        #Convolution Layer4 with 256 filters and a kernel size of 5 
#        conv4 = tf.layers.conv2d(
#        inputs = pool3, 
#        filters = 256, 
#        kernel_size = [3, 3], 
##        strides=(1, 1),
#        padding = 'same',
#        activation = tf.nn.relu )
#        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
#        pool4 = tf.layers.max_pooling2d(conv4, 2, 2)                
#        
#        
        #Convolution Layer5 with 512 filters and a kernel size of 5 
#        conv5 = tf.layers.conv2d(
#        inputs = pool4, 
#        filters = 512, 
#        kernel_size = [3, 3], 
##        strides=(1, 1),
#        padding = 'same',
#        activation = tf.nn.relu )
#        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
#        pool5 = tf.layers.max_pooling2d(conv5, 2, 2)         
        
        
        #Convolution Layer6 with 256 filters and a kernel size of 5 
#        conv6 = tf.layers.conv2d(
#        inputs = pool5, 
#        filters = 512, 
#        kernel_size = [3, 3], 
##        strides=(1, 1),
#        padding = 'same',
#        activation = tf.nn.relu )
#        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
#        pool6= tf.layers.max_pooling2d(conv6, 2, 2)         
       # Flatten the data to a 1-D vector for the fully connected layer
        
        fc1_flat = tf.contrib.layers.flatten(pool3)
        
        # Fully connected layer1 (in tf contrib folder for now)
        fc2 = tf.layers.dense(fc1_flat, 1024)#数字表示全连接层的神经元个数
        fc2 = tf.nn.relu(fc2)
        
        # Fully connected layer2        
        fc3 = tf.layers.dense(fc2, 512)
        fc3 = tf.nn.relu(fc3)
        
         #Fully connected layer3
#        fc4 = tf.layers.dense(fc3, 10)
#        fc4 = tf.nn.relu(fc4)
        
        # Apply Dropout (if is_training is False, dropout is not applied)
        # rate参数指定丢弃率
        fc5 = tf.layers.dropout(fc3, rate = 0.4, training = is_training)
        
        # Output layer, class prediction
        out = tf.layers.dense(fc5, n_classes)
        
    return out


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, n_classes, dropout, reuse = False,
                            is_training = True)
    
    logits_test = conv_net(features, n_classes, dropout, reuse = True,
                           is_training = False)

    # Predictions
    pred_classes = logits_test
    
    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions = pred_classes)

        # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.square(logits_train - labels))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step = tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels = labels, predictions = pred_classes)
#    sio.savemat('out.mat', mdict={'predictions': (pred_classes)})
    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode = mode,
        predictions = pred_classes,
#        export_outputs={'preout': pred_classes},
        loss = loss_op,
        train_op = train_op,
        eval_metric_ops = {'accuracy': acc_op})

    return estim_specs
    
# Build the Estimator
model = tf.estimator.Estimator(model_fn)

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {'y6_save_real_image': train_data1[0:15000]}, y = train_label1[0:15000],
    batch_size = batch_size, num_epochs = None, shuffle = True)
# Train the Model
model.train(input_fn, steps = num_steps)

# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {'y6_save_real_image': test_data1[0:5000]}, y = test_label1[0:5000],
    batch_size = batch_size, num_epochs = 1, shuffle = False)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)
#e2 = model.predict(input_fn)
#print("Testing Accuracy:", e['loss'])

Y_test1 = test_label1.reshape(5000)
my_array = np.empty(5000)
prediction_results = model.predict(input_fn = input_fn)
print("prediction_results: ", prediction_results)

for x, each in enumerate(prediction_results):
    my_array[x] = each


SE = (my_array - Y_test1)
#print("SE: ", SE)

MSE = sum((my_array - Y_test1)**2)/5000
print("MSE: ", MSE)


