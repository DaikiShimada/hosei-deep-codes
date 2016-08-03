# -*- coding: utf-8 -*-

from __future__ import print_function
import datetime
import six
import numpy as np
import chainer
from chainer import cuda

gpu_id = 0
if gpu_id >= 0:
    cuda.get_device(gpu_id).use()
xp = cuda.cupy if gpu_id >= 0 else np

dim = 5000
a = xp.arange(dim, dtype=xp.float32).reshape(dim,1)
at = a.T
b = np.arange(dim, dtype=np.float32).reshape(dim,1)
bt = b.T

rep = 100
# GPU
st = datetime.datetime.now()
for i in six.moves.range(rep):
    aa = a.dot(at)
ed = datetime.datetime.now()
print('GPU:', ed-st)

# CPU
st = datetime.datetime.now()
for i in six.moves.range(rep):
    bb = b.dot(bt)
ed = datetime.datetime.now()
print('CPU:', ed-st)
