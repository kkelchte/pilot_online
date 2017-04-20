"""Contains the definition for fully fully_connected control network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import batch_norm
slim = tf.contrib.slim


def depth_estim_v1(inputs,
                  num_classes=1,
                  is_training=True,
                  dropout_keep_prob=1.0,
                  min_depth=16,
                  depth_multiplier=1.0,
                  reuse=None,
                  scope='Depth_Estimate_V1'):
  end_points = {}
  # inputs of shape [batch_size, height, width, 3]
  # with tf.variable_scope(scope, 'Depth_Estimate_V1', [inputs, num_classes],
  #                        reuse=reuse) as scope:
  #   #with slim.arg_scope([slim.batch_norm, slim.dropout],
  #                       #is_training=is_training):
  #   # Final pooling and prediction
  #   
  with tf.variable_scope(scope,'Depth_Estimate_V1', reuse = reuse):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                          stride=1, padding='VALID'):
      depth = lambda d: max(int(d * depth_multiplier), min_depth)
      end_point = 'Conv'
      net = slim.conv2d(inputs, 96, [11,11], stride=4, scope=end_point)
      net = batch_norm(inputs = net, decay = 0.9997, epsilon = 0.001, is_training = is_training, scope = 'batchnorm1')
      net = slim.max_pool2d(net,[2,2],stride=2)
      # net = tf.nn.relu(net)
      end_points[end_point] = net
      end_point = 'Conv_1'
      net = slim.conv2d(net, 256, [5,5], stride=1, scope=end_point)
      net = batch_norm(inputs = net, decay = 0.9997, epsilon = 0.001, is_training = is_training, scope = 'batchnorm2')
      net = slim.max_pool2d(net,[2,2],stride=2)
      # net = tf.nn.relu(net)
      end_points[end_point] = net
      end_point = 'Conv_2'
      net = slim.conv2d(net, 384, [3,3], stride=1, scope=end_point)
      net = batch_norm(inputs = net, decay = 0.9997, epsilon = 0.001, is_training = is_training, scope = 'batchnorm3')
      # net = tf.nn.relu(net)
      end_points[end_point] = net
      end_point = 'Conv_3'
      net = slim.conv2d(net, 384, [3,3], stride=2, scope=end_point)
      net = batch_norm(inputs = net, decay = 0.9997, epsilon = 0.001, is_training = is_training, scope = 'batchnorm4')
      # net = tf.nn.relu(net)
      end_points[end_point] = net
      end_point = 'Conv_4'
      net = slim.conv2d(net, 256, [3,3], stride=1, scope=end_point)
      end_points[end_point] = net

      prelogits = tf.reshape(net, [-1, 2560])
      end_point = 'fully_connected'
      aux_logits=slim.fully_connected(prelogits, 4096, tf.nn.relu)
      end_points[end_point] = aux_logits
      end_point = 'fully_connected_1'
      # output height 55 width 74
      aux_logits=slim.fully_connected(aux_logits, 55*74, tf.nn.relu)
      aux_logits=tf.reshape(aux_logits, [-1, 55, 74])

      end_points[end_point] = aux_logits
      #with tf.variable_scope('FC'):
      with tf.variable_scope('control'):
        #with slim.arg_scope([slim.fully_connected],weights_initializer=tf.truncated_normal_initializer(stddev=0.0005),
        #  biases_initializer=tf.truncated_normal_initializer(stddev=0.0005)): 
        end_point = 'control_1'
        logits = slim.fully_connected(prelogits, 100, scope=end_point)
        end_points[end_point] = logits
        # net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
        end_point = 'control_2'
        logits = slim.fully_connected(logits, num_classes, activation_fn=tf.tanh, scope=end_point)
        end_points[end_point] = logits
  return logits, end_points

depth_estim_v1.input_size = [1,240,320,3]

def arg_scope(weight_decay=0.00004, stddev=0.1):
  """Defines the default depth_estim_v1 arg scope.
  Args:
    weight_decay: The weight decay to use for regularizing the model.
    stddev: The standard deviation of the trunctated normal weight initializer.
  Returns:
    An `arg_scope` to use for the inception v3 model.
  """
  # Set weight_decay for weights in Conv and FC layers.
  print('argscope depth estim with stddev:',stddev)
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(weight_decay), 
                      weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                      biases_initializer=tf.truncated_normal_initializer(stddev=stddev)):    
      with slim.arg_scope(
        [slim.conv2d],
        activation_fn=tf.nn.relu) as sc:
        # normalizer_fn=slim.batch_norm,
        # normalizer_params=batch_norm_params
        return sc