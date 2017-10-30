#!/usr/bin/env python
# from __future__ import print_function

# Note: in Eclipse, the environment variable CUDA_VISIBLE_DEVICES=""
# Unset to use GPU

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from datahandler import Dataset

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, dtype=tf.float32, stddev=0.05))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.05, dtype=tf.float32, shape=shape))

def conv2d_layer(input_matrix, conv_filter_size, num_input_channel, num_filter):
    weight = weight_variable([conv_filter_size, conv_filter_size, num_input_channel, num_filter])
    bias = bias_variable([num_filter])
    conv = tf.nn.conv2d(input_matrix, weight, [1, 1, 1, 1], padding='SAME') + bias
    out = tf.nn.relu(conv)
    return out

def fc_layer(input_matrix, fc_width, fc_height, use_relu=True):
    weight = weight_variable([fc_width, fc_height])
    bias = bias_variable([fc_height])
    out = tf.matmul(input_matrix, weight) + bias
    if use_relu == True:
        return tf.nn.relu(out)
    else: 
        return out
    
if __name__=='__main__':
    # Import data location
    train_dir = 'train'
    RGB = [105.354, 114.966, 123.656]
    
    print('Loading dataset from', train_dir)
    dataset = Dataset(train_dir, 0.9996, mean=RGB, shuffle=True)

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
    y_pred_cls = tf.argmax(y_pred, axis=1)
    
    # Prediction
    y_true = tf.placeholder(tf.float32, [None, 2], name='y_true')
    y_true_cls = tf.argmax(y_true, axis=1)
    
    # Evaluation
    correct_pred = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc3, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
        
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
    session.run(tf.global_variables_initializer())
            
    saver = tf.train.Saver()
    
    # Load pre-trained model if available
    session = tf.InteractiveSession()
    saver = tf.train.import_meta_graph('model/model-3300.meta')
    checkpoint = tf.train.get_checkpoint_state('model')
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(session, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not load model.")
    
    # Train
    print('Training...')
    batch_size = 10
    epoch_size = 50
    max_iteration = 100000
    feed_dict_val = {x: dataset.valid_x, y_true: dataset.valid_y_}
    log_file = open('model/log.csv', 'w')
    log_file.write('Epoch, tr_acc, vl_acc, vl_loss\n')
    for i in range(3301, max_iteration):
        print('Iteration: {} / {}'.format(i, max_iteration))
        batch_x, batch_y_ = dataset.getNextBatch(batch_size)
        feed_dict_tr = {x: batch_x, y_true: batch_y_}
        session.run(optimizer, feed_dict=feed_dict_tr)
        
        if i % epoch_size == 0:          
            saver.save(session, r'C:\Users\Echoes\Desktop\workspace\DogvsCat\model\model', global_step=i)
            
            epoch = int(i / epoch_size)
            acc = session.run(accuracy, feed_dict=feed_dict_tr)
            acc_val = session.run(accuracy, feed_dict=feed_dict_val)
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            print('Epoch', epoch, '-- Training accuracy:', acc, '-- Validation accuracy:', acc_val, '-- Validation loss:', val_loss)
            log_file.write(str(epoch) + ',' + str(acc) + ',' + str(acc_val) + ',' + str(val_loss) + '\n')
            
    saver.save(session, r'C:\Users\Echoes\Desktop\workspace\DogvsCat\model\model', global_step=i)