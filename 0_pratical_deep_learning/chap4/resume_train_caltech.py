# -*- coding: utf-8 -*-

from __future__ import print_function
import six
import csv
import os
import numpy as np
import chainer
from chainer import cuda

import masalachai

import convnet


########################
# Data Preprocesser    #
########################
def preprocess(data):
    im = masalachai.preprocesses.load_image(data['data'],
                                            target_size=(64,64))
    im = im / im.max()
    im = masalachai.preprocesses.random_rotation(im, 15)
    data['data'] = im
    return data



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
import datetime
now = datetime.datetime.today()
log_filename = 'caltech101_resume_' + now.strftime('%Y-%m-%d-%H-%M-%S') + '.json'
logger = masalachai.loggers.DictLogger('Caltec101 Log', logfile=log_filename)



########################
# Dataset setup        #
########################
# load dataset
data_dir = 'caltech101'
image_list = data_dir + '/images_list.csv'
with open(image_list, 'r') as f:
    reader = csv.reader(f)
    image_files = []
    image_categories = []
    for row in reader:
        filename = data_dir + '/' + row[0]
        category = os.path.dirname(row[0])
        image_files.append(filename)
        image_categories.append(category)
        
labels = {cat: label for label, cat in enumerate(list(set(image_categories)))}

# split Train / Test
from sklearn.cross_validation import StratifiedKFold
k = 5
skf = StratifiedKFold(image_categories, k, shuffle=False)
train_idx, test_idx =  next(iter(skf))

train_data = {'data': [image_files[i] for i in train_idx],
              'target': np.asarray([labels[image_categories[i]] for i in train_idx], dtype=np.int32)}
test_data = {'data': [image_files[i] for i in test_idx],
             'target': np.asarray([labels[image_categories[i]] for i in test_idx], dtype=np.int32)}

# construct DataFeeder
train_feeder = masalachai.DataFeeder(train_data, batchsize=16)
test_feeder = masalachai.DataFeeder(test_data, batchsize=16)

# hook preprocess
train_feeder.hook_preprocess(preprocess)
test_feeder.hook_preprocess(preprocess)



########################
# Model setup          #
########################
network = convnet.ConvNet()
chainer.serializers.load_npz('caltech_chain.npz', network)
model = masalachai.models.ClassifierModel(network)
if gpu_id >= 0:
    mdoel.to_gpu()


# Optimizer setup
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
chainer.serializers.load_npz('caltech_optimizer.npz', optimizer)



########################
# Trainer setup        #
########################
train_nitr = 1000
trainer = masalachai.trainers.SupervisedTrainer(optimizer,
                                                logger,
                                                (train_feeder,),
                                                test_feeder,
                                                gpu_id)
trainer.train(train_nitr,
              log_interval=1,
              test_interval=100,
              test_nitr=1)

# serialize Chain and Optimizer
chainer.serializers.save_npz('caltech_chain_resume.npz', network)
chainer.serializers.save_npz('caltech_optimizer_resume.npz', optimizer)

