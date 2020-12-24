from tensorflow import keras
import tensorflow as tf
import numpy as np
from generator import DataGen
import os
import matplotlib.pyplot as plt

model = keras.Sequential()
model.add(keras.layers.Dense(100, input_shape=(1,784), activation=tf.nn.sigmoid))
model.add(keras.layers.Dense(100, activation= tf.nn.relu))
model.add(keras.layers.Dense(100, activation = tf.nn.relu))
model.add(keras.layers.Dense(10, activation = tf.nn.sigmoid))
model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.01), loss = keras.losses.CategoricalCrossentropy())

print(model.summary())
directory = os.path.join('X:/MNIST/','train/')


if __name__=="__main__":
    losses = []
    epochs = 5
    for i in range(epochs):
        batch = DataGen(directory, 100, gray=False)
        for j in range(len(batch.class_list)):
            batch.class_select()
            num_batches = len(batch.files_list)//batch.batch_size
            for k in range(num_batches):
                batch_items = batch.next_batch()
                x = np.reshape(batch_items[0], (len(batch_items[0]), 1, 784))
                y = np.reshape(batch_items[1], (len(batch_items[1]), 1, int(len(batch.class_list))))
                #model.set_weights(model.get_weights())
                model_history = model.fit(x, y)
                losses.append(model_history.history['loss'])
    plt.figure(figsize= (15, 10))
    plt.plot(losses)
    plt.show()

