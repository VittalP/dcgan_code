from lib import activations
from lib.img_utils import inverse_transform
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, l2normalize, conv_with_bias
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv

from lib.config import caffe_dir
import sys
import os
import numpy as np
sys.path.append(caffe_dir + 'python')
import caffe

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

def discrim(X, w, w2, g2, b2, w3, g3, b3, w4, g4, b4, wmy):
    h = lrelu(dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2)))
    h2 = lrelu(batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2)), g=g2, b=b2))
    h3 = lrelu(batchnorm(dnn_conv(h2, w3, subsample=(2, 2), border_mode=(2, 2)), g=g3, b=b3))
    h4 = lrelu(batchnorm(dnn_conv(h3, w4, subsample=(2, 2), border_mode=(2, 2)), g=g4, b=b4))
    h4 = T.flatten(h4, 2)
    # y = sigmoid(T.dot(h4, wy))
    multi_y = softmax(T.dot(h4, wmy))
    return multi_y

def vggPool4(X, conv1_1_w, conv1_1_b, conv1_2_w, conv1_2_b, conv2_1_w, conv2_1_b, conv2_2_w, conv2_2_b, conv3_1_w, conv3_1_b, conv3_2_w, conv3_2_b, conv3_3_w, conv3_3_b, conv4_1_w, conv4_1_b, conv4_2_w, conv4_2_b, conv4_3_w, conv4_3_b):
    feat1 = T.signal.pool.pool_2d(relu(conv_with_bias(relu(conv_with_bias(X, conv1_1_w, conv1_1_b)), conv1_2_w, conv1_2_b)), ds=(2,2), ignore_border=True, mode='max')
    feat2 = T.signal.pool.pool_2d(relu(conv_with_bias(relu(conv_with_bias(feat1, conv2_1_w, conv2_1_b)), conv2_2_w, conv2_2_b)), ds=(2,2), ignore_border=True, mode='max')
    feat3 = T.signal.pool.pool_2d(relu(conv_with_bias(relu(conv_with_bias(relu(conv_with_bias(feat2, conv3_1_w, conv3_1_b)), conv3_2_w, conv3_2_b)), conv3_3_w, conv3_3_b)), ds=(2,2), ignore_border=True, mode='max')
    feat4 = T.signal.pool.pool_2d(relu(conv_with_bias(relu(conv_with_bias(relu(conv_with_bias(feat3, conv4_1_w, conv4_1_b)), conv4_2_w, conv4_2_b)), conv4_3_w, conv4_3_b)), ds=(2,2), ignore_border=True, mode='max')
    return feat4

def vgg16():
    model_def = '../models/vgg16-deploy-conv.prototxt'
    model_weights = '../models/vgg16.caffemodel'
    if not os.path.isfile(model_weights):
        print "Download VGG-16 model and place in dcgan_code/models directory."
        print "Exiting..."
        sys.exit()
    caffe.set_device(0)  # if we have multiple GPUs, pick the first one
    caffe.set_mode_gpu()
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    return net

def getVGGFeat(net, imb):
    # Data preprocessing for inputting to VGG16
    # Convert back to RGB values [0,255]
    if np.min(imb) < 0:
        imb = inverse_transform(imb, 3, 64)*255

    vgg_imb = np.zeros((imb.shape[0], 100, 100, 3))
    # VGG needs 100x100 patches
    if imb.shape[2] != 100:
        import scipy
        for ii in range(imb.shape[0]):
            vgg_img[ii,...] = scipy.misc.imresize(np.squeeze(imb[ii,::-1]), (100,100,3))
    else:
        vgg_imb = imb
    vgg_imb = vgg_imb - np.array((104.00698793,116.66876762,122.67891434))
    vgg_imb = vgg_imb.transpose((0,3,1,2))

    net.blobs['data'].data[...] = vgg_imb
    out = net.forward()
    pool4_feat = np.squeeze(out['pool4'])
    return pool4_feat
