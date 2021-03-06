#
# Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#

# This file contains functions for training a TensorFlow model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import time
from PIL import Image

import psutil as ps
from pympler import asizeof

def process_dataset():
    # Import the data
    (x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshape the data
    NUM_TRAIN = 60000
    NUM_TEST = 10000
    x_train = np.reshape(x_train, (NUM_TRAIN, 28, 28, 1))
    x_test = np.reshape(x_test, (NUM_TEST, 28, 28, 1))
    return x_train, y_train, x_test, y_test

def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=[28,28, 1]))
    model.add(tf.keras.layers.Conv2D(16, kernel_size=[3, 3], activation=tf.nn.relu, input_shape=[28, 28, 1]))
    model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2])) 
    
    model.add(tf.keras.layers.Conv2D(32, kernel_size=[3, 3], activation=tf.nn.relu))
    #model.add(tf.keras.layers.Conv2D(64, kernel_size=[6, 6], activation=tf.nn.relu))
    model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2]))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def save(model, filename):
    # First freeze the graph and remove training nodes.
    output_names = model.output.op.name
    sess = tf.keras.backend.get_session()
    frozen_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [output_names]) #zmienne -> wierzcholki grafu
    frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph) #usuwanie np:. inicjalizacje i zapis wartosci wag
    print('******************************************')
    print('******************************************')
    print('Zamrozony model: ')
    # Save the model
    for x in frozen_graph.node:
        print(x.name)
    with open(filename, "wb") as ofile:
        ofile.write(frozen_graph.SerializeToString())

#wyswietla wykres dokladnosci i straty dla danych testowych i treningowych
def draw_curves(history, key1='accuracy', ylim1=(0.8, 1.00), key2='loss', ylim2=(0.0, 1.0)):
    plt.figure(figsize=(12, 4))
    #obrazek 1
    plt.subplot(1, 2, 1)
    plt.plot(history.history[key1], "r--")
    plt.plot(history.history['val_' + key1], "g--")
    plt.ylabel('Dokladnosc')
    plt.xlabel('Epoki')
    plt.ylim(ylim1)
    plt.legend(['treningowa', 'testowa'], loc='best')
    #obrazek 2
    plt.subplot(1, 2, 2)
    plt.plot(history.history[key2], "r--")
    plt.plot(history.history['val_' + key2], "g--")
    plt.ylabel('Straty')
    plt.xlabel('Epoki')
    plt.ylim(ylim2)
    plt.legend(['treningowa', 'testowa'], loc='best')
    plt.show()

#wyswietla 40 obrazow ze zbioru testowego
def display_test_pictures(x_test, y_test):
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    plt.figure(figsize=(14, 10))
    for i in range(40):
        plt.subplot(5, 8, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_test[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[y_test[i]])
    plt.show()


def main():
    x_train, y_train, x_test, y_test = process_dataset()

    datagen = ImageDataGenerator( #do manipulacji obrazem - mala rotacja
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False)

    test_loss_avg = np.zeros(20)
    test_acc_avg = np.zeros(20)
    time_check = np.zeros(20)
    memory_check = np.zeros(20)

    time_start = time.perf_counter()
    time_stop = time.perf_counter()

    
    model = create_model()
    m_usage = asizeof.asizeof(model)
    print('--- --- --- --- Rozmiar obiektu modelu:' + str(m_usage) + ' bajtow --- --- --- ---')
    # Train the model on the data
    time_start_0 = time.perf_counter()
    history = model.fit(datagen.flow(x_train, y_train, batch_size=32), validation_data=(x_test, y_test), epochs = 18, verbose = 1)
    time_stop_0 = time.perf_counter()
    time_check_0 = time_stop_0 - time_start_0
    print('--- --- --- --- Czas trenowania: ' + str(time_check_0) + ' s --- --- --- ---')
    print("********************************************")

    display_test_pictures(x_test, y_test)

    # Evaluate the model on test data
    for x in range(20):
        time_start = time.perf_counter()
        test_loss, test_acc = model.evaluate(x_test, y_test)
        time_stop = time.perf_counter()
        test_loss_avg[x] = test_loss
        test_acc_avg[x] = test_acc
        time_check[x] = time_stop - time_start
        memory_check[x] = ps.virtual_memory()[2]
        print(str(x+1) + ". Dokladnosc: " + str(test_acc))
        print(str(x+1) + ". Strata: " + str(test_loss))
        print(str(x+1) + ". Czas: " + str(time_check[x]))
        print('Uzycie pamieci RAM w %:', ps.virtual_memory()[2])
        print("********************************************")
    
    avgC = np.average(test_acc_avg)
    avgL = np.average(test_loss_avg)
    avgT = np.average(time_check)
    avgRAM = np.average(memory_check)
    print("********************************************")
    print("Avg Dokladnosc: " + str(avgC))
    print("Avg Strata: " + str(avgL))
    print("Avg Czas: " + str(avgT) + ' s')
    print("Avg Zuzycie RAM: " + str(avgRAM) + ' %')
    
    save(model, filename="models/lenet5.pb")
    draw_curves(history, key1='acc', ylim1=(0.0, 1.0), key2='loss', ylim2=(0.0, 3.0))

if __name__ == '__main__':
    main()
