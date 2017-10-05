import sys
from os import listdir
from os.path import join

import cv2
import tensorflow as tf

if __name__=='__main__':
    # Load model
    session = tf.InteractiveSession()
    saver = tf.train.import_meta_graph('model/model-9999.meta')
    checkpoint = tf.train.get_checkpoint_state('model')
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(session, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not load model.")
        sys.exit(-1)
    
    # Start working with loaded model
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('x:0')
    y_pred = graph.get_tensor_by_name('y_pred:0')
    y_pred_cls = tf.arg_max(y_pred, dimension=1)
    
    test_dir = 'test1'
    image_files = listdir(test_dir) 
    for file in image_files:
        print('Image:', file)
        img = cv2.imread(join(test_dir, file))
        img_resized = cv2.resize(img, (224, 224))
        cv2.imshow('image', img)
        a, b = session.run([y_pred, y_pred_cls], feed_dict={x: [img_resized]})
        print('Possibility:', a)
        print('Prediction: ', b)
        
        # Break
        key = cv2.waitKey(-1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(0)    