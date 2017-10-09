from os.path import join
from os import listdir
import cv2

if __name__=='__main__':
    train_dir = 'train'
    file_names = listdir(train_dir)
    total = [0, 0, 0]
    for file in file_names:
        img = cv2.imread(join(train_dir, file))
        tmp = [0, 0, 0]
        count = 0
        for row in img:
            for pixel in row:
                tmp += pixel
                count += 1
        
        total += tmp / count       
#         print(file)
#         print(tmp, count, '-------', tmp/count)
#         print('Sum:', total)
        
    print('Average:', total / len(file_names))