import sys
sys.path.append('..')

import os
import json
from time import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.externals import joblib

import theano
import theano.sandbox.cuda
import theano.tensor as T

from lib import activations
from lib import updates
from lib import inits
from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, l2normalize
from lib.metrics import nnc_score, nnd_score
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data, center_crop, patch
from lib.config import data_dir
import lib.utils as utils
from lib import models

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
lr = 0.002       # initial learning rate for adam
desc = 'vgg_recon'
path = os.path.join(data_dir, "vc.hdf5")  # Change path to visual concepts file
tr_data, tr_stream = visual_concepts(path, ntrain=None)

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
vaX = data[patches_idx]
vaX = transform(vaX, npx)

data = tr_data.get_data(tr_handle, slice(0, tr_data.num_examples))
nvc = np.max(data[labels_idx]) # Number of visual concepts
assert nvc == 176 # Debugging code. Remove it later
nz = data[feat_l2_idx].shape[1]  # Length of the population encoding vector
ntrain = tr_data.num_examples  # # of examples to train on

model_dir = 'models/%s'%desc
samples_dir = 'samples/%s'%desc
if not os.path.exists('logs/'):
    os.makedirs('logs/')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

print "Initializing weights from scratch"
gifn = inits.Normal(scale=0.02)
difn = inits.Normal(scale=0.02)
gain_ifn = inits.Normal(loc=1., scale=0.02)
bias_ifn = inits.Constant(c=0.)

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

gen_params = [gw, gg, gb, gw2, gg2, gb2, gw3, gg3, gb3, gw4, gg4, gb4, gwx]
iter_array = range(niter)

X = T.tensor4()
Z = T.matrix()
F = T.matrix()
gF = T.matrix()

gX = models.gen(Z, *gen_params)

g_cost = T.mean(T.sum(T.pow(F-gF, 2)))

print('Here')
lrt = sharedX(lr)
g_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
g_updates = g_updater(gen_params, g_cost)
updates = g_updates

print 'COMPILING'
t = time()
_train_g = theano.function([X, Z, F, gF], g_cost, updates=g_updates)
_gen = theano.function([Z], gX)
print '%.2f seconds to compile theano functions'%(time()-t)

vis_idxs = py_rng.sample(np.arange(len(vaX)), nvis)
vaX_vis = inverse_transform(vaX[vis_idxs], nc, npx)
color_grid_vis(vaX_vis, (14, 14), 'samples/%s_etl_test.png'%desc)

sample_zmb = floatX(data[zmb_idx][vis_idxs,:])

f_log = open('logs/%s.ndjson'%desc, 'wb')
log_fields = [
    'n_epochs',
    'n_updates',
    'n_examples',
    'n_seconds',
    '1k_va_nnd',
    '10k_va_nnd',
    '100k_va_nnd',
    'g_cost'
]

vaX = vaX.reshape(len(vaX), -1)

print desc.upper()
n_updates = 0
n_check = 0
n_epochs = iter_array[0]
n_updates = 0
n_examples = 0
t = time()

vgg16_net = models.vgg16()

for epoch in iter_array:
    for data in tqdm(tr_stream.get_epoch_iterator(), total=ntrain/nbatch):
        if data[patches_idx].shape[0] != nbatch:
            continue;
        imb = data[patches_idx]
        imb = transform(imb, npx)

        zmb = floatX(data[zmb_idx])

        samples = np.asarray(_gen(zmb))
        imb_g = inverse_transform(samples, nc, npx)

        realF = models.getVGGFeat(vgg16_net, imb)
        # realF = realF / np.linalg.norm(realF)
        genF = models.getVGGFeat(vgg16_net, imb_g)
        # genF = genF / np.linalg.norm(genF)

        cost = _train_g(imb, zmb, realF, genF)
        n_updates += 1
        n_examples += len(imb)
        g_cost = float(cost[0])

    print '%.0f %.4f %.4f' % (epoch, g_cost)
    f_log.write(json.dumps(dict(zip(log_fields, log)))+'\n')
    f_log.flush()

    samples = np.asarray(_gen(sample_zmb))
    color_grid_vis(inverse_transform(samples, nc, npx), (14, 14), 'samples/%s/%d.png'%(desc, n_epochs))
    n_epochs += 1
    if n_epochs > niter:
        lrt.set_value(floatX(lrt.get_value() - lr/niter_decay))
    if n_epochs % 5 == 0:
        joblib.dump([p.get_value() for p in gen_params], 'models/%s/%d_gen_params.jl'%(desc, n_epochs))
        joblib.dump([p.get_value() for p in discrim_params], 'models/%s/%d_discrim_params.jl'%(desc, n_epochs))
