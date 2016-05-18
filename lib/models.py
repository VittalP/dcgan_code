from lib import activations
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, l2normalize
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv

import config # imports caffe_dir
import sys
import os
# sys.path.append(caffe_dir + 'python')
# import caffe

relu = activations.Rectify()
sigmoid = activations.Sigmoid()
softmax = activations.Softmax()
lrelu = activations.LeakyRectify()
tanh = activations.Tanh()
bce = T.nnet.binary_crossentropy
cce = T.nnet.categorical_crossentropy
ngf = 128 # TODO: fix this

def gen(Z, w, g, b, w2, g2, b2, w3, g3, b3, w4, g4, b4, wx):
    h = relu(batchnorm(T.dot(Z, w), g=g, b=b))
    h = h.reshape((h.shape[0], ngf*8, 4, 4))
    h2 = relu(batchnorm(deconv(h, w2, subsample=(2, 2), border_mode=(2, 2)), g=g2, b=b2))
    h3 = relu(batchnorm(deconv(h2, w3, subsample=(2, 2), border_mode=(2, 2)), g=g3, b=b3))
    h4 = relu(batchnorm(deconv(h3, w4, subsample=(2, 2), border_mode=(2, 2)), g=g4, b=b4))
    x = tanh(deconv(h4, wx, subsample=(2, 2), border_mode=(2, 2)))
    return x

def discrim(X, w, w2, g2, b2, w3, g3, b3, w4, g4, b4, wy, wmy):
    h = lrelu(dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2)))
    h2 = lrelu(batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2)), g=g2, b=b2))
    h3 = lrelu(batchnorm(dnn_conv(h2, w3, subsample=(2, 2), border_mode=(2, 2)), g=g3, b=b3))
    h4 = lrelu(batchnorm(dnn_conv(h3, w4, subsample=(2, 2), border_mode=(2, 2)), g=g4, b=b4))
    h4 = T.flatten(h4, 2)
    y = sigmoid(T.dot(h4, wy))
    multi_y = softmax(T.dot(h4, wmy))
    return y, multi_y

def vgg16(imb):
    model_def = '../models/vgg16-deploy-conv.prototxt'
    model_weights = '../models/vgg16.caffemodel'
    if not os.path.isfile(model_weights):
        print "Download VGG-16 model and place in dcgan_code/models directory."
        print "Exiting..."
        sys.exit()

    # Data preprocessing for inputting to VGG16
    # Convert back to RGB values [0,255]
    if np.min(imb) < 0:
        imb = inverse_transform(imb, 3, 64, 64)*255

    # VGG needs 100x100 patches
    if imb.shape[2] != 100:
        import scipy
        vgg_imb = np.zeros((imb.shape[0], 100, 100, 3))
        for ii in range(imb.shape[0]):
            img = scipy.misc.imresize(np.squeeze(imb[ii,::-1]), (100,100,3))
            img = img - np.array((104.00698793,116.66876762,122.67891434))
            vgg_imb[ii,...] = img.transpose((2,0,1))

    net = caffe.Net(model_def, model_weights, caffe.TEST)
    net_conv.blobs['data'].data[...] = vgg_imb
    out = net.forward()
    pool4_feat = np.squeeze(out['pool4'])
    return pool4_feat
