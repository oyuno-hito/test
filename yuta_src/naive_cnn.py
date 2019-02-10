# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import sys


def cnn(l_rate=0.001,ep=50,batch=64,conv=64):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(conv, activation=tf.nn.relu),
        #keras.layers.Dropout(0.5),
        #keras.layers.Dense(128, activation=tf.nn.relu),
        #keras.layers.Dropout(0.5),
        keras.layers.Dense(6, activation=tf.nn.softmax)
    ])
    sgd = keras.optimizers.SGD(lr=l_rate)
    adam=keras.optimizers.Adam(lr=l_rate)
    model.compile(optimizer=sgd,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    result=model.fit(train_images, train_labels,epochs=ep ,validation_data=(test_images, test_labels),verbose=0)
    return result

def load_data(path="../data/all_data.csv"):
    df=pd.read_csv(path,encoding="shift-jis")


def main():
    #argv=sys.argv
    path="../data/all_data.csv"
    boat=load_data()
    result=cnn()
    plt.plot(range(1, ep+1), result.history['acc'], label="training")
    plt.plot(range(1, ep+1), result.history['val_acc'], label="test")

if __name__=="__main__":
    main()