# -*- coding: utf-8 -*-

from __future__ import print_function
import chainer
import numpy as np
import six

import neuralnet

# define model
model = neuralnet.NeuralNet(2, 4, 1)

# data
x_data = np.asarray([[1,0], #1番目のデータ
                     [1,1], #2番目のデータ
                     [0,1], #3番目のデータ
                     [0,0]],#4番目のデータ
                     dtype=np.float32)
x = chainer.Variable(x_data)

t_data = np.asarray([[1],[0],[1],[0]], dtype=np.int32)
t = chainer.Variable(t_data)


# define optimizer
optimizer = chainer.optimizers.SGD(lr=0.8)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(0.001))

N = 1000
for i in six.moves.range(N):
    # forwarding data
    y = model(x)

    # backward and update parameters
    optimizer.update(chainer.functions.sigmoid_cross_entropy, y, t)

    # loss function for logging
    j = chainer.functions.sigmoid_cross_entropy(y, t)
    a = chainer.functions.binary_accuracy(y, t)
    print("loss=", j.data, ", accuracy=", a.data)
