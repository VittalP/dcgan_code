import sys

sys.path.append('..')

import os
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream

def visual_concepts(path, ntrain=None, nval=None, batch_size=128):

    tr_data = H5PYDataset(path, which_sets=('train',))

    ntrain = tr_data.num_examples

    tr_scheme = ShuffledScheme(examples=ntrain, batch_size=batch_size)
    tr_stream = DataStream(tr_data, iteration_scheme=tr_scheme)

    return tr_data, tr_stream