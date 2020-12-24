import os
import matplotlib.pyplot as plt
from generator import DataGen


if __name__=="__main__":
    directory_from = os.path.join('X:/MNIST/','train/')
    directory_to = os.path.join('X:/MNIST/','train_data/')
    batch_object = DataGen(directory_from, 10)

    for i in range(10):
        class_dir = os.path.join(directory_to, str(i))
        os.mkdir(os.path.join(class_dir))

        batch_object.class_select()
        batch = batch_object.next_batch()

        for k, item in enumerate(batch):
            plt.imsave(class_dir+'/'+str(k)+'.jpg', item)




