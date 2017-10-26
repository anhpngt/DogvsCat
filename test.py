from datahandler import Dataset
import numpy as np
import tensorflow as tf

if __name__=='__main__':
    log_file = open('model/log.csv', 'w')
    log_file.write('Epoch, tr_acc, vl_acc, vl_loss\n')
    for i in range(1, 20000):
        log_file.write(str(1) + ',' + str(2) + ',' + str(3) + ',' + str(i) + '\n')