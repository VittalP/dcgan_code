import sys
sys.path.append('..')

import os
import json
from time import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.externals import joblib
import scipy.misc
import theano
import theano.sandbox.cuda
import theano.tensor as T

from lib import activations
from lib import updates
from lib import inits
from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, l2normalize, conv_with_bias
from lib.metrics import nnc_score, nnd_score
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data, center_crop, patch
from lib.config import data_dir
import lib.utils as utils
from lib import models
from lib import vgg

from lib.img_utils import inverse_transform, transform

from load import visual_concepts

k = 1             # # of discrim updates for each gen update
l2 = 1e-5         # l2 weight decay
nvis = 196        # # of samples to visualize during training
b1 = 0.5          # momentum term of adam
nc = 3            # # of channels in image
nbatch = 128      # # of examples in batch
npx = 64          # # of pixels width/height of images
# nz is set later by looking at the feature vector length
# nz = 100          # # of dim for Z
ngf = 128         # # of gen filters in first conv layer
ndf = 128         # # of discrim filters in first conv layer
nx = npx*npx*nc   # # of dimensions in X
niter = 50        # # of iter at starting learning rate
niter_decay = 0   # # of iter to linearly decay learning rate to zero
lr_d = 0.0002       # initial learning rate for adam
lr_g = 0.002       # initial learning rate for adam
vggp4x = 100
desc = 'vgg_l2_multi_adv_cos_lrg'
path = os.path.join(data_dir, "vc.hdf5")  # Change path to visual concepts file
tr_data, tr_stream = visual_concepts(path, ntrain=None, batch_size=nbatch)

patches_idx = tr_stream.dataset.provides_sources.index('patches')
labels_idx = tr_stream.dataset.provides_sources.index('labels')
feat_l2_idx = tr_stream.dataset.provides_sources.index('feat_l2')
feat_orig_idx = tr_stream.dataset.provides_sources.index('feat_orig')

if "orig" in desc:
    zmb_idx = feat_orig_idx
else:
    zmb_idx = feat_l2_idx

tr_handle = tr_data.open()
data = tr_data.get_data(tr_handle, slice(0, 10000))

nvc = 176 # Visual concepts
nz = 512
ntrain = tr_data.num_examples  # # of examples to train on

model_dir = 'models/%s'%desc
samples_dir = 'samples/%s'%desc
if not os.path.exists('logs/'):
    os.makedirs('logs/')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

print "Initializing VGG weights"
vgg_keras_weights = 'models/vgg16/vgg16_weights.h5'
save_path = vgg.keras2numpy(vgg_keras_weights)
vgg_params = [sharedX(element) for element in joblib.load(save_path)]

print "Initializing weights from scratch"
gifn = inits.Normal(scale=0.02)
difn = inits.Normal(scale=0.02)
gain_ifn = inits.Normal(loc=1., scale=0.02)
bias_ifn = inits.Constant(c=0.)

# Generative model parameters
gw  = gifn((nz, ngf*8*4*4), 'gw')
gg = gain_ifn((ngf*8*4*4), 'gg')
gb = bias_ifn((ngf*8*4*4), 'gb')
gw2 = gifn((ngf*8, ngf*4, 5, 5), 'gw2')
gg2 = gain_ifn((ngf*4), 'gg2')
gb2 = bias_ifn((ngf*4), 'gb2')
gw3 = gifn((ngf*4, ngf*2, 5, 5), 'gw3')
gg3 = gain_ifn((ngf*2), 'gg3')
gb3 = bias_ifn((ngf*2), 'gb3')
gw4 = gifn((ngf*2, ngf, 5, 5), 'gw4')
gg4 = gain_ifn((ngf), 'gg4')
gb4 = bias_ifn((ngf), 'gb4')
gwx = gifn((ngf, nc, 5, 5), 'gwx')

# Discriminative model parameters
dw  = difn((ndf, nc, 5, 5), 'dw')
dw2 = difn((ndf*2, ndf, 5, 5), 'dw2')
dg2 = gain_ifn((ndf*2), 'dg2')
db2 = bias_ifn((ndf*2), 'db2')
dw3 = difn((ndf*4, ndf*2, 5, 5), 'dw3')
dg3 = gain_ifn((ndf*4), 'dg3')
db3 = bias_ifn((ndf*4), 'db3')
dw4 = difn((ndf*8, ndf*4, 5, 5), 'dw4')
dg4 = gain_ifn((ndf*8), 'dg4')
db4 = bias_ifn((ndf*8), 'db4')
dwy = difn((ndf*8*4*4, 1), 'dwy')
dwmy = difn((ndf*8*4*4, nvc*2), 'dwmy')

gen_params = [gw, gg, gb, gw2, gg2, gb2, gw3, gg3, gb3, gw4, gg4, gb4, gwx]
discrim_params = [dw, dw2, dg2, db2, dw3, dg3, db3, dw4, dg4, db4, dwmy]
iter_array = range(niter)

X = T.tensor4()
Z = T.matrix()
Y = T.matrix()

gX = models.gen(Z, *gen_params)

# Adversarial training
p_real_multi = models.discrim(X, *discrim_params)
p_gen_multi = models.discrim(gX, *discrim_params)

bce = T.nnet.binary_crossentropy
cce = T.nnet.categorical_crossentropy
# d_cost_real = bce(p_real, T.ones(p_real.shape)).mean()
# d_cost_gen = bce(p_gen, T.zeros(p_gen.shape)).mean()

d_cost_real = cce(p_real_multi, T.concatenate([Y, T.zeros((p_real_multi.shape[0], nvc))], axis=1)).mean()
d_cost_gen = cce(p_gen_multi, T.concatenate([T.zeros((p_gen_multi.shape[0], nvc)), Y], axis=1)).mean()

# g_cost_d = bce(p_gen, T.ones(p_gen.shape)).mean()
g_cost_d = cce(p_gen_multi, T.concatenate([Y, T.zeros((p_gen_multi.shape[0], nvc))], axis=1)).mean()

# VGG recon loss
gX_UP = T.nnet.abstract_conv.bilinear_upsampling(gX, ratio=2, batch_size=nbatch, num_input_channels=3)
invGX_UP = inverse_transform(gX_UP, 3, 128)*floatX(np.asarray((255)))
invGX_center, _u = theano.scan(lambda x: x[14:114, 14:114, :], sequences=invGX_UP) # Crops the center patch

# prepare data for VGG
vgg_data = invGX_center - floatX(np.asarray((104.00698793,116.66876762,122.67891434)))
vgg_data = vgg_data.dimshuffle((0,3,1,2))
gF = T.reshape(models.vggPool4(vgg_data, *vgg_params), (nbatch, nz))
g_cost_vgg_recon = T.mean(T.sum(T.pow(Z-gF, 2), axis=1))
g_cost_recon = T.mean(T.sqr(gX - X))

def cosine(A,B):
    numer = T.sum(A*B, axis=1)
    deno = T.sqrt( T.sum(A*A, axis=1) * T.sum(B*B, axis=1) )
    dis = T.mean(1. - numer/deno)
    return dis

g_cost_cosine = cosine(Z, gF)

d_cost = d_cost_real + d_cost_gen
g_cost = g_cost_d + g_cost_cosine

cost = [d_cost_real, d_cost_gen, g_cost_d, g_cost_cosine]

lrt_d = sharedX(lr_d)
lrt_g = sharedX(lr_g)
d_updater = updates.Adam(lr=lrt_d, b1=b1, regularizer=updates.Regularizer(l2=l2))
g_updater = updates.Adam(lr=lrt_g, b1=b1, regularizer=updates.Regularizer(l2=l2))
d_updates = d_updater(discrim_params, d_cost)
g_updates = g_updater(gen_params, g_cost)
updates = g_updates + d_updates

print 'COMPILING'
t = time()
_train_g = theano.function([X, Y, Z], cost, updates=g_updates)
_train_d = theano.function([X, Y, Z], cost, updates=d_updates)
_gen = theano.function([Z], gX)
print '%.2f seconds to compile theano functions'%(time()-t)

vis_idxs = py_rng.sample(np.arange(10000), nvis)
sample_zmb = floatX(data[zmb_idx][vis_idxs,:])

f_log = open('logs/%s.ndjson'%desc, 'wb')
log_fields = [
    'n_epochs',
    'n_seconds',
    'd_cost_real',
    'd_cost_gen'
    'g_cost_d',
    'g_cost_cosine'
]

print desc.upper()
n_updates = 0
n_check = 0
n_epochs = iter_array[0]
n_updates = 0
n_examples = 0
t = time()

for epoch in iter_array:
    for data in tqdm(tr_stream.get_epoch_iterator(), total=ntrain/nbatch):
        if data[patches_idx].shape[0] != nbatch:
            continue;
        # Collect batch
        imb = data[patches_idx]
        imb = transform(imb, npx)

        z = data[zmb_idx]
        zmb = floatX(z)

        labels = data[labels_idx]
        label_stack = np.array([], dtype=np.uint8).reshape(0,nvc)
        for label in labels:
            hot_vec = np.zeros((1,nvc), dtype=np.uint8)
            hot_vec[0,label-1] = 1 # labels are 1-nvc
            label_stack = np.vstack((label_stack, hot_vec))

        ymb = label_stack

        # Train
        if n_updates % (k+1) == 0:
            cost = _train_g(imb, ymb, zmb)
        else:
            cost = _train_d(imb, ymb, zmb)

        n_updates += 1
    d_cost_real = float(cost[0])
    d_cost_gen = float(cost[1])
    g_cost_d = float(cost[2])
    g_cost_cosine = float(cost[3])

    print '%.0f %f %f %f %f' % (epoch, d_cost_real, d_cost_gen, g_cost_d, g_cost_cosine)
    log = [n_epochs, time() - t, d_cost_real, d_cost_gen, g_cost_d, g_cost_cosine]
    f_log.write(json.dumps(dict(zip(log_fields, log)))+'\n')
    f_log.flush()

    samples = np.asarray(_gen(sample_zmb))
    color_grid_vis(inverse_transform(samples, nc, npx), (14, 14), 'samples/%s/%d.png'%(desc, n_epochs))
    n_epochs += 1
    if n_epochs > niter:
        lrt.set_value(floatX(lrt.get_value() - lr/niter_decay))
    if n_epochs % 5 == 0:
        joblib.dump([p.get_value() for p in gen_params], 'models/%s/%d_gen_params.jl'%(desc, n_epochs))
