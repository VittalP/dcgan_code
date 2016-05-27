import sys
sys.path.append('..')

import os
import json
from time import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from lib.img_utils import inverse_transform, transform
from lib.LogisticRegression import LogisticRegression
from lib.config import data_dir
import theano
import theano.tensor as T
from lib.theano_utils import floatX

from load import visual_concepts

l2 = 1e-5         # l2 weight decay
nc = 3            # # of channels in image
nbatch = 128      # # of examples in batch
npx = 64          # # of pixels width/height of images
nx = npx*npx*nc   # # of dimensions in X
niter = 50        # # of iter at starting learning rate
niter_decay = 0   # # of iter to linearly decay learning rate to zero
# lr = 1.0       # initial learning rate for adam
lr = float(sys.argv[2])
base_name = sys.argv[1]
desc = base_name + '_' + str(lr)
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
data = tr_data.get_data(tr_handle, slice(0, tr_data.num_examples))
nvc = np.max(data[labels_idx]) # Number of visual concepts
assert nvc == 176 # Debugging code. Remove it later
nz = data[feat_l2_idx].shape[1]  # Length of the population encoding vector
ntrain = tr_data.num_examples  # # of examples to train on

# generate symbolic variables for input (x and y represent a
# minibatch)
x = T.matrix('x')  # data, presented as rasterized images
y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

# construct the logistic regression class
classifier = LogisticRegression(input=x, n_in=nz, n_out=nvc)

# the cost we minimize during training is the negative log likelihood of
# the model in symbolic format
cost = classifier.negative_log_likelihood(y)

# compute the gradient of cost with respect to theta = (W,b)
g_W = T.grad(cost=cost, wrt=classifier.W)
g_b = T.grad(cost=cost, wrt=classifier.b)

# specify how to update the parameters of the model as a list of
# (variable, update expression) pairs.
lrt = floatX(lr)
updates = [(classifier.W, classifier.W - lrt * g_W),
           (classifier.b, classifier.b - lrt * g_b)]

train_model = theano.function(
    inputs=[x, y],
    outputs=cost,
    updates=updates
)

# compile a predictor function
predict_model = theano.function(
    inputs=[classifier.input],
    outputs=classifier.y_pred)

f_log = open('logs/%s.ndjson'%desc, 'wb')
log_fields = [
    'n_epochs',
    'train_accuracy'
]

n_updates = 0
n_check = 0
n_epochs = 0
n_updates = 0
t = time()

for epoch in range(10):
    for data in tqdm(tr_stream.get_epoch_iterator(), total=ntrain/nbatch):
        if data[patches_idx].shape[0] != nbatch:
            continue
        labels = data[labels_idx]-1
        labels = labels.reshape((labels.shape[0],))
        x_batch = floatX(data[zmb_idx])
        y_batch = labels
        batch_cost = train_model(x_batch, y_batch)

        n_updates += 1
        flag = True

    print "Testing..."
    numBatches = 0
    numCorrect = 0
    for data in tqdm(tr_stream.get_epoch_iterator(), total=ntrain/nbatch):
        if data[patches_idx].shape[0] != nbatch:
            continue
        labels = data[labels_idx] - 1
        labels = labels.reshape((labels.shape[0],))
        x_batch = floatX(data[zmb_idx])
        y_batch = labels
        y_pred = predict_model(x_batch)
        numCorrect += np.equal(y_pred, labels).sum()
        numBatches += 1

    print('epoch: %d, train_accuracy: %5f') % (epoch, numCorrect*100/(numBatches*nbatch))
    n_epochs += 1
    log = [epoch, numCorrect*100.0/(numBatches*nbatch)]
    f_log.write(json.dumps(dict(zip(log_fields, log)))+'\n')
    f_log.flush()
