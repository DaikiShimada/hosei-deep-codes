from __future__ import print_function
import numpy as np
import chainer
from chainer import cuda

gpu_id = 0
if gpu_id >= 0:
    cuda.get_device(gpu_id).use()
xp = cuda.cupy if gpu_id >= 0 else np

x = xp.asarray([[1,1,1],[2,2,2],[3,3,3]], dtype=xp.float32)
y = xp.exp(x)

vx = chainer.Variable(x)
vy = chainer.functions.exp(vx)

print('y', y)
print('vy', vy)
print('vy.data', vy.data)
