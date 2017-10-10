from datahandler import Dataset
import numpy as np

if __name__=='__main__':
    data = np.array(
    				[[[15, 15, 15], [20, 30, 40], [70, 90, 110]],
    				[[70, 90, 110], [70, 63, 110], [70, 90, 110]],
    				[[70, 50, 110], [70, 90, 200], [70, 30, 115]]])
    # a = data[:,:,0]
    print(data[:,:,0])