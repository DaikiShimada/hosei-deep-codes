# -*- coding: utf-8 -*-

import chainer

class NeuralNet(chainer.Chain):
    def __init__(self, inputs, hiddens, outputs):
        super(NeuralNet, self).__init__(
            hidden_layer = chainer.links.Linear(inputs, hiddens),
            output_layer = chainer.links.Linear(hiddens, outputs),
        )

    def __call__(self, x):
        h = self.hidden_layer(x)
        h = chainer.functions.sigmoid(h)
        out = self.output_layer(h)
        return out
