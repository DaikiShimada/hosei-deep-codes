# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import six
import numpy as np
import chainer
from chainer import cuda

import masalachai

import neuralnet


########################
# GPU setup            #
########################
gpu_id = -1 #-1: CPU, 0: GPU(0)
if gpu_id >= 0:
    cuda.get_device(gpu_id).use()
xp = cuda.cupy if gpu_id >= 0 else np


########################
# Logger setup         #
########################
logger = masalachai.Logger('XOR Log')


########################
# Dataset setup        #
########################
x_data = np.asarray([[1,0],[1,1],[0,1],[0,0],[1,0],[1,1],[0,1],[0,0]], dtype=np.float32)
t_data = np.asarray([[1],[0],[1],[0],[1],[0],[1],[0]], dtype=np.int32)

# split Train / Test
from sklearn.cross_validation import KFold
k = 4
fold = int(sys.argv[1])
skf = KFold(len(t_data), k, shuffle=False)
for i in six.moves.range(fold):
    train_idx, test_idx =  next(iter(skf))
### smater way is to use itertools.islice.

# construct DataFeeder
train_data = {'data': x_data, 'target': t_data}
train_feeder = masalachai.DataFeeder(train_data, batchsize=6)
test_feeder = masalachai.DataFeeder(train_data, batchsize=2)


########################
# Model setup          #
########################
model = masalachai.models.ClassifierModel(neuralnet.NeuralNet(2,4,1),
                                          lossfun=chainer.functions.sigmoid_cross_entropy,
                                          accuracyfun=chainer.functions.binary_accuracy)
if gpu_id >= 0:
    mdoel.to_gpu()

# Optimizer setup
optimizer = chainer.optimizers.SGD(lr=0.8)
optimizer.setup(model)


########################
# Trainer setup        #
########################
train_nitr = 300
trainer = masalachai.trainers.SupervisedTrainer(optimizer,
                                                logger,
                                                (train_feeder,),
                                                test_feeder,
                                                gpu_id)
trainer.train(train_nitr,
              log_interval=10,
              test_interval=100,
              test_nitr=1)

