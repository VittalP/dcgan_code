import sys
sys.path.append('..')

import numpy as np
from lib import models
from lib.theano_utils import floatX, sharedX
from lib.rng import py_rng, np_rng
from lib.vis import color_grid_vis
from lib.img_utils import inverse_transform, transform

from sklearn.externals import joblib

import theano
import theano.tensor as T

desc = "vc_dcgan"
model_dir = '/mnt/disk1/vittal/dcgan_code/faces/models/%s/'%desc
model_number = "25_gen_params.jl"
gen_params_np = joblib.load(model_dir + model_number)
gen_params = [sharedX(element) for element in gen_params_np]

Z = T.matrix()
gX = models.gen(Z, *gen_params)
sample_zmb = floatX(np_rng.uniform(-1., 1., size=(196, 100)))

print 'COMPILING...'
_gen = theano.function([Z], gX)
print 'Done!'

samples = np.asarray(_gen(sample_zmb))
color_grid_vis(inverse_transform(samples, nc=3, npx=64), (14, 14), '/mnt/disk1/vittal/dcgan_code/faces/samples/%s/random.png'%(desc))
