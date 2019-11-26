#simple analysis run
import numpy as np
import tensorcox as tx
import tensorflow as tf
import matplotlib.pyplot as plt

def cox_run():
    # generate random data
    np.random.seed(7)
    theta = np.asarray([1.85, -0.95, 0.25])[:, None]
    X = np.random.normal((1.1, 2.6, 10), 0.5, (1000, 3))
    T = (-((np.log(np.random.uniform(0, 1, (1000, 1))))/ (np.exp(np.matmul(X, theta))*1)))**(1/5)
    censoring = np.random.binomial(1, 0.2, (1000, 1))
    T = T - np.random.uniform(0, T) * censoring
    Surv = np.concatenate((np.repeat(0, 1000)[:, None], T, -1 * (censoring - 1)), axis=1)

    # moodel
    tf.reset_default_graph()
    X_ = tf.placeholder(tf.float64, shape=[None, X.shape[1]])
    surv_ = tf.placeholder(tf.float64, shape=[None,3])
    theta_ = tf.Variable(initial_value=tf.random_normal([X.shape[1], 1], mean=0, stddev=0.1, dtype=tf.float64))
    pred = tf.matmul(X_,  theta_)

    tcox = tx.tensorcox(surv_, pred)
    neg_ll = -tcox.loglikelihood()
    ci = tcox.concordance()

    optimizer = tf.train.AdamOptimizer(0.01).minimize(neg_ll)
    init = tf.global_variables_initializer()
    num_epochs = 3500
    with tf.Session() as sess:
        sess.run(init)
        for i in np.arange(num_epochs):
            sess.run(optimizer,  feed_dict={surv_: Surv, X_: X})
        concordance = sess.run(ci, feed_dict={surv_: Surv, X_: X})
        theta_hat = sess.run(theta_)

    bhazard = tcox.baseline_hazard(predictor=np.matmul(X, theta_hat))
    with tf.Session() as sess:
            bh, t = sess.run(bhazard,  feed_dict={surv_: Surv, X_: X})

    return (theta_hat, concordance, bh, t)

def sol():
    a, b, c, d = cox_run()
    assert np.logical_and(np.logical_and(max(a - np.asarray([1.85, -0.95, 0.25])[:, None]) < 0.1, max(b) > 0.7), c.shape[0] == d.shape[0] == 790)[0] == True
