import math
import os
import datetime

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

def kde(mu, tau, bbox=None, xlabel="", ylabel="", cmap='Blues'):
    values = np.vstack([mu, tau])
    kernel = scipy.stats.gaussian_kde(values)

    fig, ax = plt.subplots()
    ax.axis(bbox)
    ax.set_aspect(abs(bbox[1]-bbox[0])/abs(bbox[3]-bbox[2]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    xx, yy = np.mgrid[bbox[0]:bbox[1]:300j, bbox[2]:bbox[3]:300j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel(positions).T, xx.shape)
    cfset = ax.contourf(xx, yy, f, cmap=cmap)
    #plt.show()
    t = str(datetime.datetime.now())
    fname = "./figs/%s.jpg"%t
    plt.savefig(fname)


def train(train_op, x_fake, z, init, disc_loss, gen_loss, z_dim,
          n_iter=50001, n_save=2000):
  bbox = [-2, 2, -2, 2]
  batch_size = x_fake.get_shape()[0].value
  ztest = [np.random.randn(batch_size, z_dim) for i in range(10)]

  with tf.Session() as sess:
    sess.run(init)

    for i in range(n_iter):
      disc_loss_out, gen_loss_out, _ = sess.run(
          [disc_loss, gen_loss, train_op])
      if i % n_save == 0:
        print('i = %d, discriminant loss = %.4f, generator loss =%.4f' %
              (i, disc_loss_out, gen_loss_out))
        x_out = np.concatenate(
            [sess.run(x_fake, feed_dict={z: zt}) for zt in ztest], axis=0)
        kde(x_out[:, 0], x_out[:, 1], bbox=bbox)
    
def learn_mixture_of_gaussians(mode):
  print(mode)
  def x_real_builder(batch_size):
    sigma = 0.1
    skel = np.array([
        [ 1.50,  1.50],
        [ 1.50,  0.50],
        [ 1.50, -0.50],
        [ 1.50, -1.50],
        [ 0.50,  1.50],
        [ 0.50,  0.50],
        [ 0.50, -0.50],
        [ 0.50, -1.50],
        [-1.50,  1.50],
        [-1.50,  0.50],
        [-1.50, -0.50],
        [-1.50, -1.50],
        [-0.50,  1.50],
        [-0.50,  0.50],
        [-0.50, -0.50],
        [-0.50, -1.50],
    ])
    temp = np.tile(skel, (batch_size // 16 + 1,1))
    mus = temp[0:batch_size,:]
    return mus + sigma*tf.random_normal([batch_size, 2])*.2
  
  z_dim = 64
  train_op, x_fake, z, init, disc_loss, gen_loss = reset_and_build_graph(
      depth=6, width=384, x_real_builder=x_real_builder, z_dim=z_dim,
      batch_size=256, learning_rate=1e-4, mode=mode)
  n_param = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
  print("n_param=",n_param) #n_param= 1505667
  train(train_op, x_fake, z, init, disc_loss, gen_loss, z_dim)

if __name__ == "__main__":
    redirect_log_file()
    # Use plain RmsProp to optimise the GAN parameters.
    # This experiment demonstrates mode collapse in traditional GAN training.
    #with Timer() as t:
    #    learn_mixture_of_gaussians('RMS')  
    # 2.78 mins on RTX3090
    # i = 10000, discriminant loss = 0.6003, generator loss =3.2014
    with Timer() as t:
        learn_mixture_of_gaussians('SGA') 
    # 9.4848 mins
    # i = 10000, discriminant loss = 1.3856, generator loss =0.6934

    #n_param= 1505667