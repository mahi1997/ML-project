

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import tensorflow as tf

import meta
import problems


def train(sess, minimize_ops, num_epochs, num_unrolls):
  """L2L training."""
  step, update, reset, loss_last, x_last = minimize_ops

  for _ in xrange(num_epochs):
    sess.run(reset)
    for _ in xrange(num_unrolls):
      cost, final_x, unused_1, unused_2 = sess.run([loss_last, x_last,
                                                    update, step])

  return cost, final_x


class L2LTest(tf.test.TestCase):
  """Tests L2L TensorFlow implementation."""

  def testSimple(self):
    """Tests L2L applied to simple problem."""
    problem = problems.simple()
    optimizer = meta.MetaOptimizer(net=dict(
        net="CoordinateWiseDeepLSTM",
        net_options={
            "layers": (),
            # Initializing the network to zeros makes learning more stable.
            "initializer": "zeros"
        }))
    minimize_ops = optimizer.meta_minimize(problem, 20, learning_rate=1e-2)
    # L2L should solve the simple problem is less than 500 epochs.
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      cost, _ = train(sess, minimize_ops, 500, 5)
    self.assertLess(cost, 1e-5)


if __name__ == "__main__":
  tf.test.main()
