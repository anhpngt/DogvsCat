# Built-in libs
from os import listdir
from os.path import join
import multiprocessing as mp
import random

# 3rd-party libs
import cv2

class Dataset(object):
    num_processes = 5
    
    def __init__(self, data_folder, train_portion=0.9, shuffle=False):
        self.data_folder = data_folder
        self.file_names = listdir(data_folder)
        self.data_size = len(self.file_names)
        if shuffle == True:
            self.file_names = self.shuffleData(self.file_names)
        
        # Actual data, as 3d-arrays and one-hot vectors
        self.train_x, self.train_y_, self.valid_x, self.valid_y_ = self.getData(train_portion)
        
        print('Finished initialize dataset.')
        print('Data folder location: ', self.data_folder)
        print('Dataset size: ', self.data_size)
        
    def shuffleData(self, file_names):
        random.shuffle(file_names)
        return file_names
    
    def getData(self, train_portion):
        N = int(self.data_size*train_portion) # size of train set, the rest is validation set
        with mp.Manager() as manager:
            images = manager.list()
            labels = manager.list()
            processes = []
            for index in range(self.num_processes):
                p = mp.Process(target=self.getImage, args=(images, labels, index)) # Pass the list
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            return images[:N], labels[:N], images[N:], labels[:N]
    
    def getImage(self, images, labels, index):
        for i in range(index, self.data_size, self.num_processes):
            current_file_name = self.file_names[i]
            src = cv2.imread(join(self.data_folder, current_file_name))
            out_img = cv2.resize(src, (224, 224))
            images.append(out_img)
            if current_file_name[0:3] == 'cat':
                labels.append([1, 0])
            elif current_file_name[0:3] == 'dog':
                labels.append([0, 1])
            