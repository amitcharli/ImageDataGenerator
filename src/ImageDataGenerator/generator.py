import os
import matplotlib.pyplot as plt
import numpy as np


def rgb_to_gray(image):
    gray = 0.2989*image[:,:,0]+0.5870*image[:,:,1]+0.1140*image[:,:,2]
    return gray


class DataGen:
    def __init__(self, directory, n, gray = False):
        self.directory = directory
        self.class_list = os.listdir(directory)
        self.batch_size = n
        self.class_list_iter = iter(self.class_list)
        self.parent_dir = None
        self.files_list = None
        self.batch_gen = None
        self.class_selected = None
        self.gray = gray

    def class_select(self):
        self.class_selected = next(self.class_list_iter)
        self.parent_dir = os.path.join(self.directory, self.class_selected + '/')
        self.files_list = os.listdir(self.parent_dir)
        self.batch_gen = self.fix_class()

    def file_read(self):
        for file in self.files_list:
            img = plt.imread(os.path.join(self.parent_dir, file))
            img_y = np.zeros((1, len(self.class_list)))
            img_y[0, int(self.class_selected)] = int(1)
            if self.gray is True:
                yield rgb_to_gray(img), img_y
            else:
                yield img, img_y

    def fix_class(self):
        img_iter = iter(self.file_read())
        return img_iter

    def next_batch(self):
        img_batch = np.zeros((int(self.batch_size), 28, 28))
        label_batch = np.zeros((int(self.batch_size), 1 , int(len(self.class_list))))
        for i in range(self.batch_size):
            data_pack = next(self.batch_gen)
            img_batch[i,:, :] = data_pack[0]
            label_batch[i,:, :] = data_pack[1]

        return img_batch, label_batch
