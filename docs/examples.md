# Examples
In this section we show two simple examples of the Cox model and how to implement them using our algorithm.

## NCCTG Lung Cancer Data
These data are from the R survival package on survival in patients with advanced lung cancer from the North Central Cancer Treatment Group. This is a simple example with fixed covariate effects.
For details see [#]_ .

```python
# Importing modules
import sys
import os
import pandas as pd
import numpy as np
import tensorcox as tx
import tensorflow as tf
import matplotlib.pyplot as plt

# loading data from dataset folder
path = os.getcwd()
data = pd.read_csv(path + '/datasets/lung.csv', sep=';')
data = data.dropna()

# extract relevant information from data and stroing them in numpy arrays
Surv = np.asarray(data.iloc[:, 1:3].copy())
Surv =  np.concatenate((np.repeat(0, Surv.shape[0])[:, None], Surv), axis=1) # adding 0 column to confirm with intervall notation
Surv[:, -1] = Surv[:, -1] - 1 # adjusting events to be 0, 1
X = np.asarray(data.iloc[:, 3:].copy())

# model
tf.reset_default_graph()
# placeholder
X_ = tf.placeholder(tf.float64, shape=[None,X.shape[1]])
surv_ = tf.placeholder(tf.float64, shape=[None,3])
theta_ = tf.Variable(initial_value=tf.random_normal([X.shape[1], 1], mean=0, stddev=0.1, dtype=tf.float64))
pred_ = tf.matmul(X_, theta_)

# Tensorcox
tcox = tx.tensorcox(surv_, pred_)
neg_ll = -tcox.loglikelihood()
ci = tcox.concordance()

# optimizing with TF
optimizer = tf.train.AdamOptimizer(0.01).minimize(neg_ll)
init = tf.global_variables_initializer()
num_epochs = 3500
with tf.Session() as sess:
    sess.run(init)
    for i in np.arange(num_epochs):
        sess.run(optimizer,  feed_dict={surv_: Surv, X_: X})
    concordance = sess.run(ci, feed_dict={surv_: Surv, X_: X})
    theta_hat = sess.run(theta_)
theta_hat
concordance

# optimizing with TF - batches
optimizer = tf.train.AdamOptimizer(0.001).minimize(neg_ll)
init = tf.global_variables_initializer()
num_epochs = 3500
with tf.Session() as sess:
    sess.run(init)
    for i in np.arange(num_epochs):
        split = np.array_split(np.arange(Surv.shape[0]), 2)
        for jj in [0, 1]:
            sess.run(optimizer,  feed_dict={surv_: Surv[split[jj], :], X_: X[split[jj], :]})
    concordance = sess.run(ci, feed_dict={surv_: Surv, X_: X})
    theta_hat = sess.run(theta_)
theta_hat
concordance

# Breslow estimator for the baseline hazard
bhazard = tcox.baseline_hazard(predictor=np.matmul(X, theta_hat))
with tf.Session() as sess:
        t, bh= sess.run(bhazard,  feed_dict={surv_: Surv, X_: X})
plt.step(t[:], np.cumsum(bh))

```

## Stanford Heart Transplant data
These data are from the survival package in R again, but contain a time-dependent covariate effect.
As context, these are survival times of patients on the waiting list for the Stanford heart transplant program.

```python
# Importing modules
import sys
import os
import pandas as pd
import numpy as np
import tensorcox as tx
import tensorflow as tf
import matplotlib.pyplot as plt

# loading data from dataset folder
path = os.getcwd()
data = pd.read_csv(path + '/datasets/heart.csv', sep=';')

# extract relevant information from data and stroing them in numpy arrays
Surv = np.asarray(data.iloc[:, 0:3].copy())
X = np.asarray(data.iloc[:, 3:-1].copy())

# model
tf.reset_default_graph()
# placeholder
X_ = tf.placeholder(tf.float64, shape=[None,X.shape[1]])
surv_ = tf.placeholder(tf.float64, shape=[None,3])
theta_ = tf.Variable(initial_value=tf.random_normal([X.shape[1], 1], mean=0, stddev=0.1, dtype=tf.float64))
pred_ = tf.matmul(X_, theta_)

# Tensorcox
tcox = tx.tensorcox(surv_, pred_)
neg_ll = -tcox.loglikelihood()
ci = tcox.concordance()

# optimizing with TF
optimizer = tf.train.AdamOptimizer(0.001).minimize(neg_ll)
init = tf.global_variables_initializer()
num_epochs = 3000
with tf.Session() as sess:
    sess.run(init)
    for i in np.arange(num_epochs):
        if i % 500 == 0:
            print(sess.run(ci, feed_dict={surv_: Surv, X_: X}))
        sess.run(optimizer,  feed_dict={surv_: Surv, X_: X})
    concordance = sess.run(ci, feed_dict={surv_: Surv, X_: X})
    theta_hat = sess.run(theta_)
theta_hat
concordance

# optimizing with TF - batches
optimizer = tf.train.AdamOptimizer(0.001).minimize(neg_ll)
init = tf.global_variables_initializer()
num_epochs = 10000
with tf.Session() as sess:
    sess.run(init)
    for i in np.arange(num_epochs):
        split = np.split(np.arange(Surv.shape[0]), 2)
        for jj in [0, 1]:
            sess.run(optimizer,  feed_dict={surv_: Surv[split[jj], :], X_: X[split[jj], :]})
    concordance = sess.run(ci, feed_dict={surv_: Surv, X_: X})
    theta_hat = sess.run(theta_)
theta_hat
concordance

# Breslow estimator for the baseline hazard
bhazard = tcox.baseline_hazard(predictor=np.matmul(X, theta_hat))
with tf.Session() as sess:
        bh, t = sess.run(bhazard,  feed_dict={surv_: Surv, X_: X})
plt.step(t[:], np.cumsum(bh))
```

----------
