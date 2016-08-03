# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L

class ConvNet(chainer.Chain):

    def __init__(self):
        super(ConvNet, self).__init__(
                conv1 = L.Convolution2D(3, 16, 3, pad=1),
                conv2 = L.Convolution2D(16, 16, 3, pad=1),
                conv3 = L.Convolution2D(16, 16, 3, pad=1),
                fc1 = L.Linear(1024, 101),
        )

    def __call__(self, x, train=True):
        h = F.max_pooling_2d( F.relu(self.conv1(x)), 3, stride=2 )
        h = F.max_pooling_2d( F.relu(self.conv2(h)), 3, stride=2 )
        h = F.max_pooling_2d( F.relu(self.conv3(h)), 3, stride=2 )
        h = self.fc1(h)
        return h

class ConvNetwithDO(chainer.Chain):

    def __init__(self):
        super(ConvNetwithDO, self).__init__(
                conv1 = L.Convolution2D(3, 16, 3, pad=1),
                conv2 = L.Convolution2D(16, 16, 3, pad=1),
                conv3 = L.Convolution2D(16, 16, 3, pad=1),
                fc1 = L.Linear(1024, 512),
                fc2 = L.Linear(512, 101),
        )

    def __call__(self, x, train=True):
        self.train = train
        h = F.max_pooling_2d( F.relu(self.conv1(x)), 3, stride=2 )
        h = F.max_pooling_2d( F.relu(self.conv2(h)), 3, stride=2 )
        h = F.max_pooling_2d( F.relu(self.conv3(h)), 3, stride=2 )
        h = F.relu(self.fc1(h))
        h = F.dropout(h, train=self.train, ratio=0.5)
        h = self.fc2(h)
        return h

class ConvNetwithBN(chainer.Chain):

    def __init__(self):
        super(ConvNetwithBN, self).__init__(
                conv1 = L.Convolution2D(3, 16, 3, pad=1),
                bn1 = L.BatchNormalization(16),
                conv2 = L.Convolution2D(16, 16, 3, pad=1),
                bn2 = L.BatchNormalization(16),
                conv3 = L.Convolution2D(16, 16, 3, pad=1),
                bn3 = L.BatchNormalization(16),
                fc1 = L.Linear(1024, 101),
        )

    def __call__(self, x, train=True):
        self.train = train
        h = F.max_pooling_2d( F.relu(self.bn1(self.conv1(x), test=not self.train)), 3, stride=2 )
        h = F.max_pooling_2d( F.relu(self.bn2(self.conv2(h), test=not self.train)), 3, stride=2 )
        h = F.max_pooling_2d( F.relu(self.bn3(self.conv3(h), test=not self.train)), 3, stride=2 )
        h = self.fc1(h)
        return h

