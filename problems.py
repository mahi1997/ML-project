

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import sys

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import sonnet as snt
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import mnist as mnist_dataset


_nn_initializers = {
    "w": tf.random_normal_initializer(mean=0, stddev=0.01),
    "b": tf.random_normal_initializer(mean=0, stddev=0.01),
}


def simple():
  """Simple problem: f(x) = x^2."""

  def build():
    """Builds loss graph."""
    x = tf.get_variable(
        "x",
        shape=[],
        dtype=tf.float32,
        initializer=tf.ones_initializer())
    return tf.square(x, name="x_squared")

  return build


def simple_multi_optimizer(num_dims=2):
  """Multidimensional simple problem."""

  def get_coordinate(i):
    return tf.get_variable("x_{}".format(i),
                           shape=[],
                           dtype=tf.float32,
                           initializer=tf.ones_initializer())

  def build():
    coordinates = [get_coordinate(i) for i in xrange(num_dims)]
    x = tf.concat([tf.expand_dims(c, 0) for c in coordinates], 0)
    return tf.reduce_sum(tf.square(x, name="x_squared"))

  return build


def quadratic(batch_size=128, num_dims=10, stddev=0.01, dtype=tf.float32):
  """Quadratic problem: f(x) = ||Wx - y||."""

  def build():
    """Builds loss graph."""

    # Trainable variable.
    x = tf.get_variable(
        "x",
        shape=[batch_size, num_dims],
        dtype=dtype,
        initializer=tf.random_normal_initializer(stddev=stddev))

    # Non-trainable variables.
    w = tf.get_variable("w",
                        shape=[batch_size, num_dims, num_dims],
                        dtype=dtype,
                        initializer=tf.random_uniform_initializer(),
                        trainable=False)
    y = tf.get_variable("y",
                        shape=[batch_size, num_dims],
                        dtype=dtype,
                        initializer=tf.random_uniform_initializer(),
                        trainable=False)

    product = tf.squeeze(tf.matmul(w, tf.expand_dims(x, -1)))
    return tf.reduce_mean(tf.reduce_sum((product - y) ** 2, 1))

  return build


def ensemble(problems, weights=None):
  """Ensemble of problems.

  Args:
    problems: List of problems. Each problem is specified by a dict containing
        the keys 'name' and 'options'.
    weights: Optional list of weights for each problem.

  Returns:
    Sum of (weighted) losses.

  Raises:
    ValueError: If weights has an incorrect length.
  """
  if weights and len(weights) != len(problems):
    raise ValueError("len(weights) != len(problems)")

  build_fns = [getattr(sys.modules[__name__], p["name"])(**p["options"])
               for p in problems]

  def build():
    loss = 0
    for i, build_fn in enumerate(build_fns):
      with tf.variable_scope("problem_{}".format(i)):
        loss_p = build_fn()
        if weights:
          loss_p *= weights[i]
        loss += loss_p
    return loss

  return build


def _xent_loss(output, labels):
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,
                                                        labels=labels)
  return tf.reduce_mean(loss)


def mnist(layers,  
          activation="sigmoid",
          batch_size=128,
          mode="train"):
  """Mnist classification with a multi-layer perceptron."""

  if activation == "sigmoid":
    activation_op = tf.sigmoid
  elif activation == "relu":
    activation_op = tf.nn.relu
  else:
    raise ValueError("{} activation not supported".format(activation))

  # Data.
  data = mnist_dataset.load_mnist()
  data = getattr(data, mode)
  images = tf.constant(data.images, dtype=tf.float32, name="MNIST_images")
  images = tf.reshape(images, [-1, 28, 28, 1])
  labels = tf.constant(data.labels, dtype=tf.int64, name="MNIST_labels")

  # Network.
  mlp = snt.nets.MLP(list(layers) + [10],
                     activation=activation_op,
                     initializers=_nn_initializers)
  network = snt.Sequential([snt.BatchFlatten(), mlp])

  def build():
    indices = tf.random_uniform([batch_size], 0, data.num_examples, tf.int64)
    batch_images = tf.gather(images, indices)
    batch_labels = tf.gather(labels, indices)
    output = network(batch_images)
    return _xent_loss(output, batch_labels)

  return build

