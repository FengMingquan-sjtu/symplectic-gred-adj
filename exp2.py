import math
import os
import numpy as np
import sonnet as snt
import tensorflow as tf 
import kfac

import matplotlib.pyplot as plt
import scipy 
import scipy.stats

from utils import Timer, redirect_log_file
from model import *

#@title Defining the functions for the experiment

def compute_eigenvalue(sess, x, n_pts, title):
  """Computes the singular values of the covariance matrix of x.
  
  The singular values are displayed in decreasing order in a plot.
  
  Args:
    sess: a Session object.
    x: a Tensor of shape ```(batch_size, x_dim)```
    n_pts: an int; the number of points used to compute the covariance matrix
    title: a string; the title of the displayed plot
  """
  batch_size, x_dim = x.get_shape().as_list()
  # Round n_pts to the next multiple of batch_size
  n_runs = (n_pts + batch_size - 1) // batch_size
  n_pts = n_runs * batch_size
  mean = np.zeros([x_dim])
  moment = np.zeros([x_dim, x_dim])
  for _ in range(n_runs):
    x_out = sess.run(x)
    mean += np.sum(x_out, axis=0)
    moment += np.matmul(x_out.transpose(), x_out)
  mean /= n_pts
  moment /= n_pts
  mean_2 = np.expand_dims(mean, 0)
  cov = moment - np.matmul(mean_2.transpose(), mean_2)
  u, s, vh = np.linalg.svd(cov)
  plt.plot(s)
  plt.title(title)
  plt.show()


def train(train_op, x_fake, init, disc_loss, gen_loss):
  n_iter = 20001
  n_save = 2000

  with tf.Session() as sess:
    sess.run(init)

    compute_eigenvalue(sess, x_fake, 2**20, 'BEFORE TRAINING')

    for i in range(n_iter):
      sess.run(train_op)
      disc_loss_out, gen_loss_out = sess.run([disc_loss, gen_loss])
      if i % n_save == 0:
        print('i = %d, discriminant loss = %.4f, generator loss =%.4f' %
              (i, disc_loss_out, gen_loss_out))
        compute_eigenvalue(sess, x_fake, 2**15, 'iter %d' % i)

    compute_eigenvalue(sess, x_fake, 2**20, 'AFTER TRAINING')


def high_dim_gaussian_experiment(mode):
  print(mode)

  x_dim = 75
  def x_real_builder(batch_size):
    return tf.random_normal([batch_size, x_dim])

  train_op, x_fake, unused_z, init, disc_loss, gen_loss = reset_and_build_graph(
      depth=2, width=200, x_real_builder=x_real_builder, z_dim=200,
      batch_size=64, learning_rate=2e-4, mode=mode)

  n_param = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
  print("n_param=",n_param) #n_param= 151076
  train(train_op, x_fake, init, disc_loss, gen_loss)

if __name__ == "__main__":
    # Use plain RmsProp to optimise the GAN parameters.
    # This experiment demonstrate how traditional GAN training fails to learn all
    # the directions of a high dimensional Gaussian.
    #redirect_log_file()
    with Timer() as t:
        high_dim_gaussian_experiment('RMS') 
    # 1.1928 min.
    #i = 20000, discriminant loss = 1.4067, generator loss =0.6321

    #with Timer() as t:
    #    high_dim_gaussian_experiment('SGA')
    #i = 20000, discriminant loss = 1.3662, generator loss =0.7044
    #Elapsed: 1.8474 min.

    #n_param= 151076