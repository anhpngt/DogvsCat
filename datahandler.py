# Built-in libraries
from os import listdir
from os.path import join
import multiprocessing as mp
import random

# 3rd-party libraries
import cv2

class Dataset(object):
    num_processes = 5
    
    def __init__(self, data_folder, train_portion=0.99, mean=[0., 0., 0.], shuffle=True):
        self.data_folder = data_folder
        self.file_names = listdir(data_folder)
        self.data_size = len(self.file_names)
        self.mean = mean
        self.training_size = int(self.data_size*train_portion) # size of train set, the rest is validation set
        self.batch_pos = 0  # Position to get batch
        if shuffle == True:
            self.list_train_x, self.list_valid_x = self.shuffleData(self.file_names)
        else:
            N = self.training_size
            self.list_train_x = self.file_names[:N]
            self.list_valid_x = self.file_names[N:]
            
        self.valid_x, self.valid_y_ = self.getData(self.list_valid_x)
#         print(self.valid_x)
        print('Finished initializing dataset.')
        print('Data folder location:', self.data_folder)
        print('Dataset size:', self.data_size)
        print('Validation set size:', len(self.valid_x))
        
    def shuffleData(self, file_names):
        N = self.training_size
        random.shuffle(file_names)
        return file_names[:N], file_names[N:]
        
    def getNextBatch(self, batch_size):
        next_pos = self.batch_pos + batch_size
        batch_x, batch_y_ = self.getData(self.list_train_x[self.batch_pos:next_pos])
        if next_pos >= self.training_size:
            next_pos = 0
        self.batch_pos = next_pos
        return batch_x, batch_y_
    
    def getData(self, images_list):
        with mp.Manager() as manager:
            images = manager.list()
            labels = manager.list()
            processes = []
            for index in range(self.num_processes):
                p = mp.Process(target=self.getImage, args=(images_list, images, labels, index)) # Pass the list
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            return images[:], labels[:]
        
    def getImage(self, images_list, images, labels, index):
        for i in range(index, len(images_list), self.num_processes):
            current_file_name = images_list[i]
            src = cv2.imread(join(self.data_folder, current_file_name))
            out_img = cv2.resize(src, (224, 224))
            
            images.append(list(out_img))
            if current_file_name[0:3] == 'cat':
                labels.append([1, 0])
            elif current_file_name[0:3] == 'dog':
                labels.append([0, 1])
                