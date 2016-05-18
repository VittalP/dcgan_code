import sys
import os
sys.path.append('..')

import numpy as np
from lib import models
from lib.config import data_dir
from lib.theano_utils import floatX, sharedX
from lib.img_utils import transform
from sklearn.externals import joblib

import theano
import theano.tensor as T
from tqdm import tqdm

from load import visual_concepts

dcgan_root = "/mnt/disk1/vittal/dcgan_code/visual_concepts/"

desc = "vcgan_orig_multi"
model_dir = dcgan_root + '/models/%s/'%desc
model_number = "25_discrim_params.jl"
discrim_params_np = joblib.load(model_dir + model_number)
discrim_params = [sharedX(element) for element in discrim_params_np]
X = T.tensor4()
Y = T.matrix()
YMULTI = T.matrix()
YHAT = T.matrix()
YHAT_MULTI = T.matrix()

dX = models.discrim(X, *discrim_params)
print 'COMPILING...'
_dis = theano.function([X], dX)
print 'Done!'

# Data processing
path = os.path.join(data_dir, "vc.hdf5")
tr_data, tr_stream = visual_concepts(path, ntrain=None)
patches_idx = tr_stream.dataset.provides_sources.index('patches')
labels_idx = tr_stream.dataset.provides_sources.index('labels')

nbatch = 128; npx = 64; nvc = 176;
total = 0; incorrect_real = 0; incorrect_multi_real = 0;

for data in tqdm(tr_stream.get_epoch_iterator(), total=tr_data.num_examples/nbatch):
    imb = data[patches_idx]
    imb = transform(imb, npx)

    labels = data[labels_idx]
    label_stack = np.array([], dtype=np.uint8).reshape(0,nvc)
    for label in labels:
        hot_vec = np.zeros((1,nvc), dtype=np.uint8)
        hot_vec[0,label-1] = 1 # labels are 1-nvc
        label_stack = np.vstack((label_stack, hot_vec))

    ymb = label_stack
    p12 = [np.asarray(element) for element in _dis(imb)]
    p_yn = p12[0]; p_multi = p12[1];
    false_indices = p_yn < 0.5;
    incorrect_real = incorrect_real + np.sum(false_indices, dtype=np.float32)

    multi_pred = np.argmax(p_multi, axis=1)
    diff = (labels - 1).reshape(labels.shape[0],1) - multi_pred.reshape(labels.shape[0],1)
    incorrect_indices = np.asarray(diff != 0)
    incorrect_multi_real = incorrect_multi_real + np.sum(incorrect_indices, dtype=np.float32)
    total = total + labels.shape[0]

d_cost_real = float(incorrect_real)/total
d_cost_multi_real = float(incorrect_multi_real)/total

print("Y/N error: %3f, Multi error: %3f" % (float(d_cost_real), float(d_cost_multi_real)))
