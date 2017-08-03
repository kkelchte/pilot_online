"""Contains the definition for fully fully_connected control network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def fc_control_v1(inputs,
                 num_classes=1,
                 is_training=True,
                 dropout_keep_prob=1.0,
                 reuse=None,
                  scope='FC_ControlV1'):
  end_points = {}
  with tf.variable_scope(scope, 'FC_ControlV1', [inputs, num_classes],
                         reuse=reuse) as scope:
    #with slim.arg_scope([slim.batch_norm, slim.dropout],
                        #is_training=is_training):
    # Final pooling and prediction
    with tf.variable_scope('FC'): 
      net = slim.fully_connected(inputs, 100, activation_fn=tf.nn.relu, scope='layer1')
      end_points['Layer1'] = net
      net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
      logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='layer2')
      # logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                             # normalizer_fn=None, scope='layer2')
      end_points['Layer2'] = logits
        
  return logits, end_points

fc_control_v1.input_size = (64)

def fc_control_v1_arg_scope(weight_decay=0.00004,
                            stddev=0.1,
                            batch_norm_var_collection='moving_vars'):
  """Defines the default fc_control_v1 arg scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    stddev: The standard deviation of the trunctated normal weight initializer.
    batch_norm_var_collection: The name of the collection for the batch norm
      variables.

  Returns:
    An `arg_scope` to use for the inception v3 model.
  """
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': 0.9997,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
      # collection containing update_ops.
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      # collection containing the moving mean and moving variance.
      'variables_collections': {
          'beta': None,
          'gamma': None,
          'moving_mean': [batch_norm_var_collection],
          'moving_variance': [batch_norm_var_collection],
      }
  }

  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
        activation_fn=tf.nn.relu,
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params) as sc:
      return sc

