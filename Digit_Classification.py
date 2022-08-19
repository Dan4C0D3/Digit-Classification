# An attempt to create a neural network to classify digits from the MNIST dataset (28x28 pixels)

import struct, numpy as np, json
from matplotlib.font_manager import json_load

# Import the MNIST Test and Train datasets
TEST_DATA_FILENAME = 'd:/Personal Projects/OCR_Machine_Learning_Project/Data/images-test-10k.idx3-ubyte'
TEST_LABELS_FILENAME = 'd:/Personal Projects/OCR_Machine_Learning_Project/Data/labels-test-10k.idx1-ubyte'
TRAIN_DATA_FILENAME = 'd:/Personal Projects/OCR_Machine_Learning_Project/Data/images-train.idx3-ubyte'
TRAIN_LABELS_FILENAME = 'd:/Personal Projects/OCR_Machine_Learning_Project/Data/labels-train.idx1-ubyte'

# Creates 2 matrices, one containing the image data (image content (28x28) X number of images) --> 3D array
#   and one containing the labels of each image (label X number of images) --> 2D array
# https://notebook.community/rasbt/pattern_classification/data_collecting/reading_mnist
def read_data(filename_X, filename_Y, max_num=-1):
    with open(filename_Y, 'rb') as file_Y:
        magic, n = struct.unpack('>II', file_Y.read(8))
        if max_num != -1: n = max_num
        labels = np.fromfile(file_Y, count=max_num, dtype=np.uint8).reshape(n, 1)

    with open(filename_X, "rb") as file_X:
        magic, n, rows, cols = struct.unpack('>IIII', file_X.read(16))
        if max_num != -1: n = max_num
        images = np.fromfile(file_X, count=max_num*(rows*cols), dtype=np.uint8).reshape(n, rows, cols)

    return images, labels


def init_parameters():
    # Creates a random 8x8 filter
    Filter = np.random.randn(8,8)
    print("Filter:\n", Filter)
    W1, W2 = np.random.randn(10, 9), np.random.randn(10, 10)
    B0, B1, B2 = np.random.randn(1, 1), np.random.randn(10, 1), np.random.randn(10, 1)
    
    return Filter, B0, W1, B1, W2, B2
    
def forward_propagation(Filter, B0, W1, B1, W2, B2, X):
    feature_map = create_feature_map(Filter, X) + B0
    ReLU_feature_map = ReLU(feature_map)
    max_pool_map = max_pooling(ReLU_feature_map)
    input_layer = flatten(max_pool_map).reshape((1,9))
    
    Z1 = W1.dot(input_layer.T) + B1
    print("Z1:\n", Z1)
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + B2
    print("Z2:\n", Z2)
    A2 = softMax(Z2)  # gives nan values 
    
    return Z1, A1, Z2, A2
    
    
# Creates a feature map using an 8x8 filter and a stepsize of 4
def create_feature_map(Filter, X):
    feature_map = np.zeros(shape=(int((X.shape[0]/4) - 1), int((X.shape[1]/4) - 1)))
    for x in range(0, X.shape[0] - 4, 4):
        for y in range(0, X.shape[1] - 4, 4):
            feature_map[int(x/4), int(y/4)] = np.sum(np.multiply(Filter, X[x:x+8, y:y+8]))
    print("Feature Map:\n", feature_map)
    return feature_map

def ReLU(matrix):
    """for number_x, value_x in enumerate(matrix):
        for number_y, value_y in enumerate(value_x):
            if value_y < 0:
                matrix[number_x, number_y] = 0"""
    ReLU = np.maximum(0, matrix)
    print("RELU:\n", ReLU)
    return ReLU

# Return a new matrix containing the maximum value of each 2x2 areas in a given matrix
def max_pooling(matrix):
    stepsize = 2
    gridsize = 2   # uses a 2x2 area to compute max
    
    max_pool = np.zeros(shape=(int(matrix.shape[0]/stepsize), int(matrix.shape[1]/stepsize)))
    for x in range(0, matrix.shape[0], stepsize):
        for y in range(0, matrix.shape[1], stepsize):
            max_pool[int(x/stepsize), int(y/stepsize)] = np.amax(matrix[x:x+gridsize, y:y+gridsize])
    print("Max Pool:\n", max_pool)
    return max_pool

def flatten(matrix):
    flat = matrix.flatten()
    print("Flat:\n", flat)
    return flat

def softMax(Z):
    max_ = np.max(Z)
    adjusted = np.exp(Z - max_)
    softmax = adjusted / np.sum(adjusted)
    print("SoftMax:\n", softmax)
    return softmax


def main():
    np.set_printoptions(linewidth=500)
    X_train, Y_train = read_data(TRAIN_DATA_FILENAME, TRAIN_LABELS_FILENAME)
    print(X_train[0])
    
    Filter, B0, W1, B1, W2, B2 = init_parameters()
    guess = forward_propagation(Filter, B0, W1, B1, W2, B2, X_train[0])
    
    
    """Map = feature_map(X_train[0])
    ReLU_map = ReLU(Map)
    max_pool = max_pooling(ReLU_map)
    flat = flatten(max_pool)
    print("Flat:\n", flat)"""

if __name__ == '__main__':
    main()