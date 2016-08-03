# -*- coding: utf-8 -*-

from __future__ import print_function
import datetime
import six
import multiprocessing

def f(x, y):
    total = 0
    for a in x:
        for b in y:
            total += a*b
    return total


SIZE = 10000
x = [i for i in six.moves.range(SIZE)]
y = [i for i in six.moves.range(SIZE)]


st = datetime.datetime.now()

# multiprocessing version
jobs = 8
pool = multiprocessing.Pool(jobs)
result_list = [pool.apply_async(f, (x[i*SIZE/jobs:(i+1)*SIZE/jobs], y)) for i in six.moves.range(jobs)]
result_mult = sum([r.get() for r in result_list])
pool.close()

ed = datetime.datetime.now()
t_m = ed - st



st = datetime.datetime.now()

# single version
result_single = f(x, y)

ed = datetime.datetime.now()
t_s = ed - st


print('Multi process', t_m)
print('Single process', t_s)
