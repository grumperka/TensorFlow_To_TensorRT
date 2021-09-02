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

# This sample uses a UFF MNIST model to create a TensorRT Inference Engine
from random import randint
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import psutil as ps
from pympler import asizeof

import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit

import tensorrt as trt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class ModelData(object): 
    MODEL_FILE = "lenet5.uff"
    INPUT_NAME ="input_1"
    INPUT_SHAPE = (1, 28, 28)
    OUTPUT_NAME = "dense_1/Softmax" #warstwa wyjściowa, zwracająca prawdopodobieństwa

def build_engine(model_file):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = common.GiB(1)
        # Parse the Uff Network - uniwersalny parser
        parser.register_input(ModelData.INPUT_NAME, ModelData.INPUT_SHAPE) #określenie danych wejściowych
        parser.register_output(ModelData.OUTPUT_NAME) #określenie danych wyjściowych
        parser.parse(model_file, network)
        # Build and return an engine.
        return builder.build_cuda_engine(network)

# Loads 1 test case into the provided pagelocked_buffer.
def load_normalized_test_case(data_paths, pagelocked_buffer, case_num):
    [test_case_path] = common.locate_files(data_paths, [str(case_num)]) 
    # Flatten the image into a 1D array, normalize, and copy to pagelocked memory.
    img = np.array(Image.open(test_case_path)).ravel() #konwersja obrazu do macierzy
    np.copyto(pagelocked_buffer, 1.0 - img / 255.0) 
    return case_num

#Ladowanie nazw obrazow
def load_files_names():
    array_list = os.listdir('test')
    return array_list

#Utworzenie listy do obliczenia straty
def make_label_array(x):
    array_label = np.zeros(10)
    array_label[x] = 1
    return array_label

def main():
    data_paths, _ = common.find_sample_data(description="Runs an MNIST network using a UFF model file", subfolder="mnist")
    model_path = os.environ.get("MODEL_PATH") or os.path.join(os.path.dirname(__file__), "models")
    model_file = os.path.join(model_path, ModelData.MODEL_FILE)
    data_to_displayC = np.zeros(10000) #wartosci dokladnosci
    data_to_displayL = np.zeros(10000) #wartosci straty
    memory_check = np.zeros(10000)

    array_files = load_files_names()
    print('Ilosc danych do testowania: ' + str(len(array_files)))
    i = 0

    time_check = np.zeros(10000) #czas przetwarzania obrazu przez konwolucyjna siec neuronowa

    with build_engine(model_file) as engine:
        # Build an engine, allocate buffers and create a stream.
        # For more information on buffer allocation, refer to the introductory samples.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        m_engine = asizeof.asizeof(engine)
        print('--- --- --- --- Rozmiar obiektu silnika:' + str(m_engine) + ' bajtow --- --- --- ---')
        with engine.create_execution_context() as context:
            for x in array_files:
                case_num = load_normalized_test_case(data_paths, pagelocked_buffer=inputs[0].host, case_num = x)
                number = int(x[0])
                array_label = make_label_array(number)
                # For more information on performing inference, refer to the introductory samples.
                # The common.do_inference function will return a list of outputs - we only have one in this case.
                time_start = time.perf_counter()
                [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
                time_stop = time.perf_counter()
                memory_check[i] = ps.virtual_memory()[2]
                time_check[i] = time_stop - time_start
                pred = np.argmax(output) #cyfra o najwiekszym prawdopodobienstwie
                confidence = np.max(output) #wartosc najwiekszego prawdopodobienstwa
                data_to_displayC[i] = confidence
                loss_conf_crossEntropy = - np.sum(np.log(output)*array_label) #obliczanie straty przez funkcje cross entropy
                data_to_displayL[i] = loss_conf_crossEntropy
                i=i+1
                
    #wyswietlanie
    avgC = np.average(data_to_displayC)
    avgL = np.average(data_to_displayL)
    avgRAM = np.average(memory_check)
    print("Avg Dokladnosc: " + str(avgC))
    print("Avg Strata: " + str(avgL))
    print("Czas: " + str(sum(time_check)) + ' s')
    print("Avg Zuzycie RAM: " + str(avgRAM) + ' %')

if __name__ == '__main__':
    main()
