import os
import matplotlib.pylab as plt
from generator import DataGen
import tensorflow as tf

directory = os.path.join('X:/MNIST/','train_data/')

if __name__=="__main__":
    batch = DataGen(directory, 2)

    for i in range(len(batch.class_list)):
        batch.class_select()
        print(batch.class_selected)
        batch_items = batch.next_batch()

        for item in batch_items:
            plt.imshow(item)
            #plt.show()
            img = tf.image.rgb_to_grayscale(item)
            print(img.shape)


