from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
from sklearn.externals import joblib
import sys, os

def VGG_16_Keras(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def keras2numpy(weights_path):
    if not os.path.exists(weights_path):
        print("VGG_16 Keras weights not found. Please download it from the follwing link: https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing")
        print("Exiting...")
        sys.exit(0)

    print "Collecting VGG_16 weights from keras model..."
    model = VGG_16_Keras(weights_path)
    print "...done!"

    relevant_layers = ['conv1_1_w', 'conv1_1_b', 'conv1_2_w', 'conv1_2_b', 'conv1_3_w', 'conv1_3_b', 'conv2_1_w', 'conv2_1_b', 'conv2_2_w', 'conv2_2_b', 'conv2_3_w', 'conv2_3_b', 'conv3_1_w', 'conv3_1_b', 'conv3_2_w', 'conv3_2_b', 'conv3_3_w', 'conv3_3_b', 'conv4_1_w', 'conv4_1_b', 'conv4_2_w', 'conv4_2_b', 'conv4_3_w', 'conv4_3_b']
    num_layers = len(relevant_layers)
    count = 0
    weights = []
    for layer in model.layers:
        if 'convolution' not in layer.name:
            continue
        if count >= num_layers/2:
            break
        weights = weights + [w for w in layer.get_weights()]
        count += 1
    print len(weights)
    print len(relevant_layers)
    assert len(weights) == len(relevant_layers)
    weight_dir = 'models/vgg16/'
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    print "Saving numpy version of vgg16 weights at " + weight_dir + "vgg16.jl"
    joblib.dump(weights, weight_dir + 'vgg16.jl')
    print "Done saving"

if __name__ == "__main__":

    if len(sys.argv) == 1:
        print "Path to VGG16 weights needs to be provided."
        print "Exiting"
        sys.exit(0)

    weights_path = sys.argv[1]
    keras2numpy(weights_path)
