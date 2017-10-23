from os.path import join
from os import listdir
import numpy as np
import cv2

if __name__=='__main__':
    train_dir = 'train_reduced'
    file_names = listdir(train_dir)
    total = np.array([0, 0, 0], dtype=np.int64)
    file_count = 0
    pixel_count = 0
    for file in file_names:
        file_count += 1
        img = np.array(cv2.imread(join(train_dir, file)))
        tmp = np.array([np.sum(img[:,:,0]), np.sum(img[:,:,1]), 
                        np.sum(img[:,:,2])], dtype=np.int32)
        count = img.shape[0] * img.shape[1]
        pixel_count += count
        total = total + tmp
 
        print('File {}: {}'.format(file_count, file))
        # print('Shape: {} x {}'.format((img.shape[0], img.shape[1])))
        print('{} -------- {}'.format(tmp, count))
        print('Sum: {}'.format(total))
         
    print('Average: {}'.format(total / pixel_count))