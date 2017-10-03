#!/usr/bin/env python
from os import listdir
from os.path import join
import multiprocessing as mp
import random

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import cv2
import numpy as np
import tensorflow as tf

from datahandler import Dataset

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, dtype=tf.float32, stddev=0.05))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.05, dtype=tf.float32, shape=shape))

def conv2d_layer(input, conv_filter_size, num_input_channel, num_filter):
    weight = weight_variable([conv_filter_size, conv_filter_size, num_input_channel, num_filter])
    bias = bias_variable([num_filter])
    conv = tf.nn.conv2d(input, weight, [1, 1, 1, 1], padding='SAME') + bias
    out = tf.nn.relu(conv)
    return out

def fc_layer(input, fc_width, fc_height, use_relu=True):
    weight = weight_variable([fc_width, fc_height])
    bias = bias_variable([fc_height])
    out = tf.matmul(input, weight) + bias
    if use_relu == True:
        return tf.nn.relu(out)
    else: 
        return out
    
if __name__=='__main__':
    # Import data location
    train_dir = 'train_reduced'
    test_dir = 'test1'
    
    print('Loading dataset from ', train_dir)
    dataset = Dataset(train_dir, 0.9, shuffle=True)
    
#     for i in range(0, dataset.data_size):
#         img = dataset.train_x[i]
#         print('label: ', dataset.train_y_[i])
#         cv2.imshow('img', img)
#           
#         # break
#         key = cv2.waitKey(-1) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('p'):
#             cv2.waitKey(0)    
    
    # Layer network
    print('Creating network...')
    session = tf.Session()
    x = tf.placeholder(tf.float32, [None, 224, 224, 3], name='x')
    
    conv1_1 = conv2d_layer(x, 3, 3, 64)
    conv1_2 = conv2d_layer(conv1_1, 3, 64, 64)
    pool1 = tf.nn.max_pool(conv1_2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    
    conv2_1 = conv2d_layer(pool1, 3, 64, 128)
    conv2_2 = conv2d_layer(conv2_1, 3, 128, 128)
    pool2 = tf.nn.max_pool(conv2_2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    
    conv3_1 = conv2d_layer(pool2, 3, 128, 256)
    conv3_2 = conv2d_layer(conv3_1, 3, 256, 256)
    conv3_3 = conv2d_layer(conv3_2, 3, 256, 256)
    pool3 = tf.nn.max_pool(conv3_3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    
    conv4_1 = conv2d_layer(pool3, 3, 256, 512)
    conv4_2 = conv2d_layer(conv4_1, 3, 512, 512)
    conv4_3 = conv2d_layer(conv4_2, 3, 512, 512)
    pool4 =  tf.nn.max_pool(conv4_3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    
    conv5_1 = conv2d_layer(pool4, 3, 512, 512)
    conv5_2 = conv2d_layer(conv5_1, 3, 512, 512)
    conv5_3 = conv2d_layer(conv5_2, 3, 512, 512)
    pool5 = tf.nn.max_pool(conv5_3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    
    shape_pool5 = int(np.prod(pool5.get_shape()[1:]))
    pool5_flatten = tf.reshape(pool5, [-1, shape_pool5])
    fc1 = fc_layer(pool5_flatten, shape_pool5, 4096)
    fc2 = fc_layer(fc1, 4096, 4096)
    fc3 = fc_layer(fc2, 4096, 2, use_relu=False)
    
    y_pred = tf.nn.softmax(fc3, name='y_pred')
    y_pred_cls = tf.arg_max(y_pred, dimension=1)
    
    # Prediction
    y_true = tf.placeholder(tf.float32, [None, 2], name='y_true')
    y_true_cls = tf.arg_max(y_true, dimension=1)
    
    # Evaluation
    correct_pred = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc3, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
    session.run(tf.global_variables_initializer())
        
    saver = tf.train.Saver()
    
    # Train
    print('Training...')
    batch_size = 50
    feed_dict_val = {x: dataset.valid_x, y_true: dataset.valid_y_}
    for i in range(0, 3000):
        batch_x, batch_y_ = dataset.getNextBatch(batch_size)
        feed_dict_tr = {x: batch_x, y_true: batch_y_}
        session.run(optimizer, feed_dict=feed_dict_tr)
        
        if i % 100 == 0:
            saver.save(session, 'dogvscat_model')
            
            acc = session.run(accuracy, feed_dict=feed_dict_tr)
            acc_val = session.run(accuracy, feed_dict=feed_dict_val)
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            print('Epoch ', i / 100, '\t-- Training accuracy: ', acc, '\t-- Validation accuracy: ', acc_val,
                  '\t-- Validation loss: ', val_loss)