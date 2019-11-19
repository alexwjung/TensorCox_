import sys
import os
import tensorflow as tf
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt


tf.enable_eager_execution()


file = open('/Users/alexwjung/Desktop/lung.csv', 'r')
lung = pd.read_csv(file, sep=',',header=0)
file.close()

time = np.asarray(lung['time'])[:, None]
events = (np.asarray(lung['status']) - 1)[:, None]
time = tf.convert_to_tensor(time)
events = tf.convert_to_tensor(events)

i0 = tf.constant(0)
t0 =  tf.cast(tf.constant(0), tf.int64)[None]
s0 =  tf.cast(tf.constant(1), tf.float64)[None]
ti = tf.boolean_mask(time, tf.equal(events, 1), name='event_times')[:, None]
order = tf.nn.top_k(-ti[:, 0], tf.shape(ti)[0])[1]
ti = tf.gather(ti, order)
cond = lambda i, tt, ss: i < ti.shape[0]
funct = lambda i, tt, ss : [i+1, tf.concat([tt, ti[i]], axis=0),
 tf.concat([ss, (1 - (tf.reduce_sum(tf.cast(tf.equal(ti, ti[i]), tf.float64))/tf.reduce_sum(tf.cast(tf.greater(time, ti[i]), tf.float64))))[None]], axis=0)]
i, tt, ss = tf.while_loop(
    cond, funct, loop_vars=[i0, t0, s0],
     shape_invariants=[i0.get_shape(), tf.TensorShape([None]),
      tf.TensorShape([None])], parallel_iterations=1)

plt.step(tt, np.cumprod(ss))
