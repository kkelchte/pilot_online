
import tensorflow as tf
import os
# import tensorflow.contrib.losses as losses
import tensorflow.contrib.slim as slim
#from tensorflow.contrib.slim.nets import inception
import inception
import fc_control
import nfc_control
import depth_estim
import mobile_net

from tensorflow.contrib.slim import model_analyzer as ma
from tensorflow.python.ops import variables as tf_variables

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

import numpy as np
import math



FLAGS = tf.app.flags.FLAGS

# Weight decay of inception network
tf.app.flags.DEFINE_float("weight_decay", 0.00004, "Weight decay of inception network")
# Std of uniform initialization
tf.app.flags.DEFINE_float("init_scale", 0.0005, "Std of uniform initialization")
# Base learning rate
tf.app.flags.DEFINE_boolean("random_learning_rate", False, "Use sampled learning rate from UL(10**-4, 1)")
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Start learning rate.")
tf.app.flags.DEFINE_float("depth_weight", 1, "Define the weight applied to the depth values in the loss relative to the control loss.")
tf.app.flags.DEFINE_float("odom_weight", 1, "Define the weight applied to the odometry values in the loss relative to the control loss.")
# Specify where the Model, trained on ImageNet, was saved.
tf.app.flags.DEFINE_string("model_path", 'mobilenet_small', "Specify where the Model, trained on ImageNet, was saved: PATH/TO/vgg_16.ckpt, inception_v3.ckpt or ")
# tf.app.flags.DEFINE_string("model_path", '/users/visics/kkelchte/tensorflow/models', "Specify where the Model, trained on ImageNet, was saved: PATH/TO/vgg_16.ckpt, inception_v3.ckpt or ")
# Define the initializer
tf.app.flags.DEFINE_string("checkpoint_path", 'mobilenet_small', "Specify the directory of the checkpoint of the earlier trained model.")
tf.app.flags.DEFINE_boolean("continue_training", False, "Specify whether the training continues from a checkpoint or from a imagenet-pretrained model.")
tf.app.flags.DEFINE_boolean("grad_mul", True, "Specify whether the weights of the final tanh activation should be learned faster.")
tf.app.flags.DEFINE_float("grad_mul_weight", 0.001, "Specify the amount the gradients of the cnn should be less applied.")
tf.app.flags.DEFINE_boolean("freeze", False, "Specify whether feature extracting network should be frozen and only the logit scope should be trained.")
tf.app.flags.DEFINE_integer("exclude_from_layer", 8, "In case of training from model (not continue_training), specify up untill which layer the weights are loaded: 5-6-7-8. Default 8: only leave out the logits and auxlogits.")
tf.app.flags.DEFINE_boolean("plot_activations", False, "Specify whether the activations are weighted.")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5, "Specify the probability of dropout to keep the activation.")
tf.app.flags.DEFINE_integer("clip_grad", 0, "Specify the max gradient norm: default 0, recommended 4.")
tf.app.flags.DEFINE_string("optimizer", 'adadelta', "Specify optimizer, options: adam, adadelta, gradientdescent, rmsprop")
tf.app.flags.DEFINE_boolean("plot_histograms", False, "Specify whether to plot histograms of the weights.")
tf.app.flags.DEFINE_boolean("feed_previous_action", False, "Feed previous action as concatenated feature for odom prediction layers.")
tf.app.flags.DEFINE_boolean("concatenate_depth", False, "Add depth prediction of 2 last frames for odometry prediction.")
tf.app.flags.DEFINE_boolean("concatenate_odom", False, "Add odom prediction of 2 last frames for control prediction.")
tf.app.flags.DEFINE_integer("odom_hidden_units", 50, "Define the number of hidden units in the odometry decision layer.")
tf.app.flags.DEFINE_string("odom_loss", 'mean_squared', "absolute_difference or mean_squared or huber")
tf.app.flags.DEFINE_string("depth_loss", 'huber', "absolute_difference or mean_squared or huber")
tf.app.flags.DEFINE_string("no_batchnorm_learning", True, "In case of no batchnorm learning, are the batch normalization params (alphas and betas) not learned")
tf.app.flags.DEFINE_boolean("extra_control_layer", False, "Add an extra hidden control layer with 50 units in case of n_fc.")

"""
Build basic NN model
"""
class Model(object):
 
  def __init__(self,  session, input_size, output_size, prefix='model', device='/gpu:0', bound=1, depth_input_size=(55,74)):
    '''initialize model
    '''
    self.sess = session
    self.output_size = output_size
    self.bound=bound
    self.input_size = input_size
    self.input_size[0] = None
    self.depth_input_size = depth_input_size
    self.prefix = prefix
    self.device = device

    np.random.seed(FLAGS.random_seed)
    tf.set_random_seed(FLAGS.random_seed)
    if FLAGS.random_learning_rate:
      # self.lr = 10**np.random.uniform(-4,0)
      self.lr = 10**np.random.uniform(-3,0)
    else:
      self.lr = FLAGS.learning_rate 
    print 'learning rate: ', self.lr    
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.define_network()
    
    if not FLAGS.continue_training and not FLAGS.evaluate:
      if FLAGS.model_path[0]!='/':
        checkpoint_path = os.path.join(os.getenv('HOME'),'tensorflow/log',FLAGS.model_path)
        # checkpoint_path = '/home/klaas/tensorflow/log/'+FLAGS.model_path
      else:
        checkpoint_path = FLAGS.model_path
      list_to_exclude = ["global_step"]
      if FLAGS.exclude_from_layer <= 7:
        list_to_exclude.extend(["InceptionV3/Mixed_7a", "InceptionV3/Mixed_7b", "InceptionV3/Mixed_7c"])
      if FLAGS.exclude_from_layer <= 6:
        list_to_exclude.extend(["InceptionV3/Mixed_6a", "InceptionV3/Mixed_6b", "InceptionV3/Mixed_6c", "InceptionV3/Mixed_6d", "InceptionV3/Mixed_6e"])
      if FLAGS.exclude_from_layer <= 5:
        list_to_exclude.extend(["InceptionV3/Mixed_5a", "InceptionV3/Mixed_5b", "InceptionV3/Mixed_5c", "InceptionV3/Mixed_5d"])
      list_to_exclude.extend(["InceptionV3/Logits", "InceptionV3/AuxLogits"])
      list_to_exclude.append("MobilenetV1/control")
      list_to_exclude.append("MobilenetV1/aux_depth")
      list_to_exclude.append("concatenated_feature")
      list_to_exclude.append("control")
      list_to_exclude.append("aux_odom")
      list_to_exclude.append("lstm_control")
      list_to_exclude.append("lstm_output")

      if FLAGS.network == 'depth':
        # control layers are not in pretrained depth checkpoint
        list_to_exclude.extend(['Depth_Estimate_V1/control/control_1/weights',
          'Depth_Estimate_V1/control/control_1/biases',
          'Depth_Estimate_V1/control/control_2/weights',
          'Depth_Estimate_V1/control/control_2/biases'])
      #print list_to_exclude
      variables_to_restore = slim.get_variables_to_restore(exclude=list_to_exclude)
      # remap only in case of using Toms original network
      if FLAGS.network == 'depth' and checkpoint_path == os.path.join(os.getenv('HOME'),'tensorflow/log','depth_net_checkpoint'):
        variables_to_restore = {
          'Conv/weights':slim.get_unique_variable('Depth_Estimate_V1/Conv/weights'),
          'Conv/biases':slim.get_unique_variable('Depth_Estimate_V1/Conv/biases'),
          'NormMaxRelu1/beta':slim.get_unique_variable('Depth_Estimate_V1/batchnorm1/beta'),
          'NormMaxRelu1/moving_mean':slim.get_unique_variable('Depth_Estimate_V1/batchnorm1/moving_mean'),
          'NormMaxRelu1/moving_variance':slim.get_unique_variable('Depth_Estimate_V1/batchnorm1/moving_variance'),
          'Conv_1/weights': slim.get_unique_variable('Depth_Estimate_V1/Conv_1/weights'),
          'Conv_1/biases':slim.get_unique_variable('Depth_Estimate_V1/Conv_1/biases'),
          'NormMaxRelu2/beta':slim.get_unique_variable('Depth_Estimate_V1/batchnorm2/beta'),
          'NormMaxRelu2/moving_mean':slim.get_unique_variable('Depth_Estimate_V1/batchnorm2/moving_mean'),
          'NormMaxRelu2/moving_variance':slim.get_unique_variable('Depth_Estimate_V1/batchnorm2/moving_variance'),
          'Conv_2/weights':slim.get_unique_variable('Depth_Estimate_V1/Conv_2/weights'),
          'Conv_2/biases':slim.get_unique_variable('Depth_Estimate_V1/Conv_2/biases'),
          'NormRelu1/beta':slim.get_unique_variable('Depth_Estimate_V1/batchnorm3/beta'),
          'NormRelu1/moving_mean':slim.get_unique_variable('Depth_Estimate_V1/batchnorm3/moving_mean'),
          'NormRelu1/moving_variance':slim.get_unique_variable('Depth_Estimate_V1/batchnorm3/moving_variance'),
          'Conv_3/weights':slim.get_unique_variable('Depth_Estimate_V1/Conv_3/weights'),
          'Conv_3/biases':slim.get_unique_variable('Depth_Estimate_V1/Conv_3/biases'),
          'NormRelu2/beta':slim.get_unique_variable('Depth_Estimate_V1/batchnorm4/beta'),
          'NormRelu2/moving_mean':slim.get_unique_variable('Depth_Estimate_V1/batchnorm4/moving_mean'),
          'NormRelu2/moving_variance':slim.get_unique_variable('Depth_Estimate_V1/batchnorm4/moving_variance'),
          'Conv_4/weights':slim.get_unique_variable('Depth_Estimate_V1/Conv_4/weights'),
          'Conv_4/biases':slim.get_unique_variable('Depth_Estimate_V1/Conv_4/biases'),
          'fully_connected/weights':slim.get_unique_variable('Depth_Estimate_V1/fully_connected/weights'),
          'fully_connected/biases':slim.get_unique_variable('Depth_Estimate_V1/fully_connected/biases'),
          'fully_connected_1/weights':slim.get_unique_variable('Depth_Estimate_V1/fully_connected_1/weights'),
          'fully_connected_1/biases':slim.get_unique_variable('Depth_Estimate_V1/fully_connected_1/biases')
          }
    else: #If continue training
      variables_to_restore = slim.get_variables_to_restore()
      if FLAGS.checkpoint_path[0]!='/':
        checkpoint_path = os.path.join(os.getenv('HOME'),'tensorflow/log',FLAGS.checkpoint_path)
      else:
        checkpoint_path = FLAGS.checkpoint_path
    
    if 'fc_control' in FLAGS.network and not FLAGS.continue_training:
      init_assign_op = None
    else:
      # get latest folder out of training directory if there is no checkpoint file
      if not os.path.isfile(checkpoint_path+'/checkpoint'):
        checkpoint_path = checkpoint_path+'/'+[mpath for mpath in sorted(os.listdir(checkpoint_path)) if os.path.isdir(checkpoint_path+'/'+mpath) and not mpath[-3:]=='val' and os.path.isfile(checkpoint_path+'/'+mpath+'/checkpoint')][-1]
      print('checkpoint: {}'.format(checkpoint_path))
      init_assign_op, init_feed_dict = slim.assign_from_checkpoint(tf.train.latest_checkpoint(checkpoint_path), variables_to_restore)
  
    # create saver for checkpoints
    self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5, max_to_keep=10)
    
    # Add the loss function to the graph.
    self.define_loss()

    
    # Define the training op based on the total loss
    self.define_train()
    
    # Define summaries
    self.build_summaries()
    
    init_all=tf_variables.global_variables_initializer()
    self.sess.run([init_all])
    if init_assign_op != None:
      self.sess.run([init_assign_op], init_feed_dict)
      print('Successfully loaded model from:', checkpoint_path)  
    # import pdb; pdb.set_trace()
  
  def define_network(self):
    '''build the network and set the tensors
    '''
    with tf.device(self.device):
      self.inputs = tf.placeholder(tf.float32, shape = self.input_size)
      if FLAGS.network=='inception':
        ### initializer is defined in the arg scope of inception.
        ### need to stick to this arg scope in order to load the inception checkpoint properly...
        ### weights are now initialized uniformly 
        with slim.arg_scope(inception.inception_v3_arg_scope(weight_decay=FLAGS.weight_decay,
                             stddev=FLAGS.init_scale)):
          #Define model with SLIM, second returned value are endpoints to get activations of certain nodes
          self.outputs, self.endpoints, self.auxdepth = inception.inception_v3(self.inputs, num_classes=self.output_size, 
            is_training=(not FLAGS.evaluate), dropout_keep_prob=FLAGS.dropout_keep_prob)  
      elif FLAGS.network=='mobile' or FLAGS.network=='mobile_small' or FLAGS.network=='mobile_medium':
        if FLAGS.network=='mobile_small': depth_multiplier = 0.25 
        elif FLAGS.network=='mobile_medium': depth_multiplier = 0.5 
        else : depth_multiplier = 1 
        if FLAGS.n_fc:
          self.inputs = tf.placeholder(tf.float32, shape = (self.input_size[0],self.input_size[1],self.input_size[2],FLAGS.n_frames*self.input_size[3]))
          self.prev_action=tf.placeholder(tf.float32, shape=(None, 1))
          def feature_extract(is_training=True):
            features = []
            for i in range(FLAGS.n_frames):
              _, endpoints = mobile_net.mobilenet_v1(self.inputs[:,:,:,i*3:(i+1)*3], num_classes=self.output_size, 
                is_training=is_training, reuse= (i!=0 and is_training) or not is_training, depth_multiplier=depth_multiplier)
              features.append(tf.squeeze(endpoints['AvgPool_1a'],[1,2]))
              if FLAGS.concatenate_depth:
                features.append(endpoints['aux_depth_enc'])
                # features.append(endpoints['aux_depth_fc_1'])
            with tf.variable_scope('concatenated_feature', reuse=not is_training): 
              features=tf.concat(features, axis=1)
              # features = tf.squeeze(features,[1,2])
              # print 'features ',features
              if is_training:
                features = slim.dropout(features, keep_prob=FLAGS.dropout_keep_prob, scope='Dropout_1b')  
            with tf.variable_scope('aux_odom', reuse=not is_training):
              aux_odom_input = tf.concat([features,self.prev_action], axis=1) if FLAGS.feed_previous_action else features
              aux_odom_logits = slim.fully_connected(aux_odom_input, FLAGS.odom_hidden_units, tf.nn.relu, normalizer_fn=None, scope='Fc_aux_odom')
              aux_odom = slim.fully_connected(aux_odom_logits, 6, None, normalizer_fn=None, scope='Fc_aux_odom_1')
              # aux_odom = slim.fully_connected(aux_odom_logits, 4, None, normalizer_fn=None, scope='Fc_aux_odom_1')
            with tf.variable_scope('control', reuse=not is_training):  
              control_input = features if not FLAGS.concatenate_odom else tf.concat([features,aux_odom_logits],axis=1)
              if FLAGS.extra_control_layer:
                control_input = slim.fully_connected(control_input, 50, tf.nn.relu, normalizer_fn=None, scope='H_fc_control')
              outputs = slim.fully_connected(control_input, 1, None, normalizer_fn=None, scope='Fc_control')
            aux_depth = endpoints['aux_depth_reshaped']
            return outputs, aux_depth, aux_odom, endpoints
          with slim.arg_scope(mobile_net.mobilenet_v1_arg_scope(is_training= True, weight_decay=FLAGS.weight_decay,
                               stddev=FLAGS.init_scale)):
            self.outputs, self.aux_depth, self.aux_odom, self.endpoints = feature_extract(True)
          with slim.arg_scope(mobile_net.mobilenet_v1_arg_scope(is_training= False, weight_decay=FLAGS.weight_decay,
                               stddev=FLAGS.init_scale)):
            self.controls, self.pred_depth, self.pred_odom, _ = feature_extract(False)
          
        elif FLAGS.lstm:
          def define_lstm(is_training=True, inputs_ph=[]):
            # define and CNN+LSTM with mobilenet
            # returns the lstm cell, a batch of outputs, a list of batches of depth predictions for all timesteps
            # self.inputs = tf.placeholder(tf.float32, shape = (self.input_size[0],None,self.input_size[1],self.input_size[2],self.input_size[3]))
            with tf.variable_scope("lstm_control", reuse=not is_training):
              def lstm():
                # lstm_cell = tf.contrib.rnn.LSTMCell(FLAGS.lstm_hiddensize, forget_bias=0)
                lstm_cell = tf.nn.rnn_cell.LSTMCell(FLAGS.lstm_hiddensize, forget_bias=0)
                # lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=FLAGS.dropout_keep_prob if is_training else 1)
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=FLAGS.dropout_keep_prob if is_training else 1)
                return lstm_cell
              # cell = tf.contrib.rnn.MultiRNNCell([lstm(), lstm()])
              cell = tf.nn.rnn_cell.MultiRNNCell([lstm(), lstm()])
              initial_state = cell.zero_state(FLAGS.batch_size if is_training else 1, tf.float32)
              # state = self.init_state = tf.placeholder(tf.float32, shape = (FLAGS.batch_size if is_training else 1, 2*2*FLAGS.lstm_hiddensize))
              state = initial_state
              outputs = []
              # states = []
              aux_depths = []
              for time_step in range(FLAGS.num_steps if is_training else 1):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                _, endpoints = mobile_net.mobilenet_v1(inputs_ph[:,time_step,:,:], num_classes=self.output_size, 
                  is_training=is_training, reuse=(time_step!=0 and is_training) or not is_training,
                  dropout_keep_prob=FLAGS.dropout_keep_prob, depth_multiplier=depth_multiplier)
                aux_depths.append(endpoints['aux_depth_reshaped'])
                (output, state) = cell(tf.reshape(endpoints['AvgPool_1a'], (FLAGS.batch_size if is_training else 1, -1)), state)
                outputs.append(output)
                # states.append(state)
            final_state = state
            with tf.variable_scope("lstm_output", reuse=not is_training):
              # concatenate in 0 direction: axis 0: [batch_at_time1, batch_at_time2, batch_at_time3, ...], axis 1: outputs
              outputs = tf.reshape(tf.concat(outputs, axis=0),(-1, FLAGS.lstm_hiddensize))
              aux_depths = tf.reshape(tf.concat(aux_depths,axis=0),(-1,55,74))
              weights = tf.get_variable("weights",[FLAGS.lstm_hiddensize, self.output_size])        
              biases = tf.get_variable('biases', [self.output_size])
              # [?, ouput_size]=[?, hidden_size]*[hidden_size, output_size]
              # with ? = num_steps * batch_size (b0t0, b0t1, ..., b1t0, b1t1, ...)
              outputs = tf.matmul(outputs, weights) + biases
            return cell, outputs, final_state, initial_state, aux_depths
            # return cell, outputs, final_state, initial_state, aux_depths
          with slim.arg_scope(mobile_net.mobilenet_v1_arg_scope(is_training=True, weight_decay=FLAGS.weight_decay,
                               stddev=FLAGS.init_scale)):
            self.inputs = tf.placeholder(tf.float32, shape = (FLAGS.batch_size,FLAGS.num_steps,self.input_size[1],self.input_size[2],self.input_size[3]))
            self.lstm, self.outputs, _, self.initial_state, self.aux_depth = define_lstm(True, self.inputs)
          with slim.arg_scope(mobile_net.mobilenet_v1_arg_scope(is_training=False, weight_decay=FLAGS.weight_decay,
                               stddev=FLAGS.init_scale)):
            self.inputs_eva = tf.placeholder(tf.float32, shape=(1,1,self.input_size[1],self.input_size[2],self.input_size[3]))
            self.lstm_eva, self.controls, self.state, self.initial_state_eva, self.pred_depth = define_lstm(False, self.inputs_eva)
          # self.lstm_eva, self.controls, self.state, self.initial_state_eva, self.pred_depth = define_lstm(False, self.inputs_eva)
          # import pdb; pdb.set_trace()
        
        else:
          if FLAGS.auxiliary_odom : raise IOError('Odometry cant be predicted when there is no n_fc')  
          #Define model with SLIM, second returned value are endpoints to get activations of certain nodes
          with slim.arg_scope(mobile_net.mobilenet_v1_arg_scope(is_training=True, weight_decay=FLAGS.weight_decay,
                           stddev=FLAGS.init_scale)):
            self.outputs, self.endpoints = mobile_net.mobilenet_v1(self.inputs, num_classes=self.output_size, 
              is_training=True, dropout_keep_prob=FLAGS.dropout_keep_prob, depth_multiplier=depth_multiplier)
            self.aux_depth = self.endpoints['aux_depth_reshaped']
          with slim.arg_scope(mobile_net.mobilenet_v1_arg_scope(is_training=False, weight_decay=FLAGS.weight_decay,
                           stddev=FLAGS.init_scale)):
            self.controls, _ = mobile_net.mobilenet_v1(self.inputs, num_classes=self.output_size, 
              is_training=False, reuse = True, depth_multiplier=depth_multiplier)
            self.pred_depth = _['aux_depth_reshaped']
      elif FLAGS.network == 'fc_control': #in case of fc_control
        with slim.arg_scope(fc_control.fc_control_v1_arg_scope(weight_decay=FLAGS.weight_decay,
                            stddev=FLAGS.init_scale)): 
          self.outputs, self.endpoints = fc_control.fc_control_v1(self.inputs, num_classes=self.output_size, 
            is_training=(not FLAGS.evaluate), dropout_keep_prob=FLAGS.dropout_keep_prob)
          self.controls, _ = fc_control.fc_control_v1(self.inputs, num_classes=self.output_size, 
            is_training=False, dropout_keep_prob=FLAGS.dropout_keep_prob, reuse=True)
      elif FLAGS.network == 'nfc_control': #in case of fc_control
        self.inputs = tf.placeholder(tf.float32, shape = (None, self.input_size[1]*4))
        with slim.arg_scope(nfc_control.fc_control_v1_arg_scope(weight_decay=FLAGS.weight_decay,
                            stddev=FLAGS.init_scale)): 
          self.outputs, self.endpoints = nfc_control.fc_control_v1(self.inputs, num_classes=self.output_size, 
            is_training=(not FLAGS.evaluate), dropout_keep_prob=FLAGS.dropout_keep_prob)
          self.controls, _ = nfc_control.fc_control_v1(self.inputs, num_classes=self.output_size, 
            is_training=False, dropout_keep_prob=FLAGS.dropout_keep_prob, reuse=True)
      elif FLAGS.network=='depth':
        with slim.arg_scope(depth_estim.arg_scope(weight_decay=FLAGS.weight_decay, stddev=FLAGS.init_scale)):
          # Define model with SLIM, second returned value are endpoints to get activations of certain nodes
          self.outputs, self.endpoints = depth_estim.depth_estim_v1(self.inputs, num_classes=self.output_size, is_training=True)
          self.aux_depth = self.endpoints['fully_connected_1']
          self.controls, _ = depth_estim.depth_estim_v1(self.inputs, num_classes=self.output_size, is_training=False, reuse = True)
          self.pred_depth = _['fully_connected_1']
      else:
        raise NameError( '[model] Network is unknown: ', FLAGS.network)
      if FLAGS.plot_histograms:
        for v in tf.global_variables():
          tf.summary.histogram(v.name.split(':')[0], v)
      if(self.bound!=1 or self.bound!=0):
        # self.outputs = tf.mul(self.outputs, self.bound) # Scale output to -bound to bound
        self.outputs = tf.multiply(self.outputs, self.bound) # Scale output to -bound to bound

  # def get_init_state(self, eva=True):
  #   # if eva: return tf.zeros([1, FLAGS.lstm_hiddensize]).eval(session=self.sess)
  #   # if eva: return self.lstm_eva.zero_state(1, tf.float32)
  #   if eva: return self.lstm_eva.zero_state(1, tf.float32).eval(session=self.sess)
  #   # else: return tf.zeros([FLAGS.batch_size, FLAGS.lstm_hiddensize]).eval(session=self.sess)
  #   # else: return self.lstm.zero_state(FLAGS.batch_size, tf.float32)
  #   else: return self.lstm.zero_state(FLAGS.batch_size, tf.float32).eval(session=self.sess)

  def define_loss(self):
    '''tensor for calculating the loss
    '''
    with tf.device(self.device):
      self.targets = tf.placeholder(tf.float32, [None, self.output_size])
      # self.loss = losses.mean_squared_error(tf.clip_by_value(self.outputs,1e-10,1.0), self.targets)
      if not FLAGS.rl or FLAGS.auxiliary_ctr:
        self.loss = tf.losses.mean_squared_error(self.outputs, self.targets)
      if FLAGS.freeze:
        FLAGS.depth_weight=0
        FLAGS.odom_weight=0
      if FLAGS.auxiliary_depth or FLAGS.rl:
        self.depth_targets = tf.placeholder(tf.float32, [None,55,74])
      if FLAGS.auxiliary_depth:
        weights = FLAGS.depth_weight*tf.cast(tf.greater(self.depth_targets, 0), tf.float32) # put loss weight on zero where depth is negative.        
        if FLAGS.depth_loss == 'mean_squared':
          self.depth_loss = tf.losses.mean_squared_error(self.aux_depth,self.depth_targets,weights=weights)
        elif FLAGS.depth_loss == 'absolute_difference':
          self.depth_loss = tf.losses.absolute_difference(self.aux_depth,self.depth_targets,weights=weights)
        elif FLAGS.depth_loss == 'huber':
          self.depth_loss = tf.losses.huber_loss(self.aux_depth,self.depth_targets,weights=weights)
        else :
          raise 'Depth loss is unknown: {}'.format(FLAGS.depth_loss)
      if FLAGS.auxiliary_odom:
        self.odom_targets = tf.placeholder(tf.float32, [None,6])
        # self.odom_targets = tf.placeholder(tf.float32, [None,4])
        if FLAGS.odom_loss == 'absolute_difference':
          self.odom_loss = tf.losses.absolute_difference(self.aux_odom,self.odom_targets,weights=FLAGS.odom_weight)
        elif FLAGS.odom_loss == 'mean_squared':
          self.odom_loss = tf.losses.mean_squared_error(self.aux_odom,self.odom_targets,weights=FLAGS.odom_weight)
        elif FLAGS.odom_loss == 'huber':
          self.odom_loss = tf.losses.huber_loss(self.aux_odom,self.odom_targets,weights=FLAGS.odom_weight)  
        else :
          raise 'Odom loss is unknown: {}'.format(FLAGS.odom_loss)
      self.total_loss = tf.losses.get_total_loss()
      
  def define_train(self):
    '''applying gradients to the weights from normal loss function
    '''
    with tf.device(self.device):
      # Specify the optimizer and create the train op:
      if FLAGS.optimizer == 'adam':    
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr) 
      elif FLAGS.optimizer == 'adadelta':
        self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr) 
      elif FLAGS.optimizer == 'gradientdescent':
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr) 
      elif FLAGS.optimizer == 'rmsprop':
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr) 
      else:
        raise IOError('Model: Unknown optimizer.')
      # Create the train_op and scale the gradients by providing a map from variable
      # name (or variable) to a scaling coefficient:
      gradient_multipliers = {}
      if FLAGS.grad_mul:        
        mobile_variables = [v for v in tf.global_variables() if (v.name.find('Adadelta')==-1 and v.name.find('BatchNorm')==-1 and v.name.find('Adam')==-1  and v.name.find('aux_depth')==-1 and v.name.find('aux_odom')==-1  and v.name.find('control')==-1)]
        for v in mobile_variables:
          # print v.name
          gradient_multipliers[v.name]=FLAGS.grad_mul_weight
      
      if FLAGS.no_batchnorm_learning:
        batchnorm_variables = [v for v in tf.global_variables() if v.name.find('BatchNorm')!=-1]
        for v in batchnorm_variables:
          # print v.name
          gradient_multipliers[v.name]=0
      # if FLAGS.freeze:
      #   global_variables = [v for v in tf.global_variables() if (v.name.find('Adadelta')==-1 and v.name.find('BatchNorm')==-1)]
      #   control_variables = [v for v in global_variables if v.name.find('control')!=-1]   # changed logits to control
      #   print('Only training control variables: ',[v.name for v in control_variables])      
      #   self.train_op = slim.learning.create_train_op(self.total_loss, self.optimizer, global_step=self.global_step, variables_to_train=control_variables, clip_gradient_norm=FLAGS.clip_grad)



      if not FLAGS.rl:
        self.train_op = slim.learning.create_train_op(self.total_loss, self.optimizer, global_step=self.global_step, gradient_multipliers=gradient_multipliers, clip_gradient_norm=FLAGS.clip_grad)
        # self.train_op = slim.learning.create_train_op(self.total_loss, self.optimizer, global_step=self.global_step, gradient_multipliers=gradient_multipliers, clip_gradient_norm=FLAGS.clip_grad)
      else:
        grads_and_vars_output = self.optimizer.compute_gradients(self.outputs, tf.trainable_variables())
        # for each sample in batch get costs spread over actionspace (-1:1)~74width pixels
        costs = 1/(tf.reduce_mean(self.depth_targets, axis=1)+0.01)-1/5
        # importance_weights with mu=self.outputs and sigma=1
        sigma=1
        # costs [batch_size, 74] * normal_weights [74, 1]
        depth_horizontal_size = 74
        # depth_horizontal_size = tf.shape(self.depth_targets)[2].eval(session=self.sess)
        action_range = list(reversed([2*x/depth_horizontal_size-1 for x in range(0,depth_horizontal_size)]))
        summed_cost = 0
        actions=self.outputs
        # actions=tf.unpack(tf.placeholder(tf.float32,[FLAGS.batch_size,1]))
        # actions=tf.stack(self.outputs)
        for b in range(FLAGS.batch_size):
          # weight gaussian over importance of that possible action
          normal_weights = [1/(sigma*tf.sqrt(2*np.pi))*tf.exp(-.5*((x-actions[b])/sigma)**2) for x in action_range]
          # Average over batch
          summed_cost=summed_cost+tf.matmul([costs[b]],normal_weights)[0]/FLAGS.batch_size
          # weighted_costs.append(tf.matmul(costs[b],normal_weights))
        self.cost_to_go = summed_cost
        # grads_and_vars = [(weighted_costs[i]*gvt[0],gvt[1]) for i,gvt in enumerate(grads_and_vars)]
        grads_and_vars_output = [(summed_cost*gvt[0] if gvt[0]!=None else None,gvt[1]) for gvt in grads_and_vars_output ]
        grads_and_vars_loss = self.optimizer.compute_gradients(self.total_loss, tf.trainable_variables())
        # TODO: assert that the tensors for which gradients are defined are the same !
        grads_and_vars=[]
        assert len(grads_and_vars_output) == len(grads_and_vars_loss), StandardError('gradient computations of trainable variables to output and loss are not the same!')
        for i in range(len(grads_and_vars_output)):
          assert grads_and_vars_output[i][1]==grads_and_vars_loss[i][1], StandardError('Gradients and variables of outputs and loss are not corresponding!')
          grads_and_vars.append((grads_and_vars_output[i][0]+grads_and_vars_loss[i][0] if grads_and_vars_output[i][0] != None and grads_and_vars_loss[i][0]!=None else None,
                    grads_and_vars_output[i][1]))
        # grads_and_vars = grads_and_vars_output       
        self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
      
        
  def forward(self, inputs, states=[], auxdepth=False, auxodom=False, prev_action=[], targets=[], target_depth=[], target_odom=[]):
    '''run forward pass and return action prediction
    '''
    tensors = [self.controls]
    if FLAGS.lstm:
      # if len(states)==0 : feed_dict={self.inputs_eva: inputs}
      if len(states)==0 : 
        states = self.sess.run(self.initial_state_eva)
      # else: feed_dict={self.inputs_eva: inputs, self.initial_state_eva: states}
      feed_dict={self.inputs_eva: inputs, self.initial_state_eva: states}
      tensors.append(self.state)
    else:
      feed_dict={self.inputs: inputs}
    if auxdepth and FLAGS.auxiliary_depth: 
      tensors.append(self.pred_depth)
    if auxodom and FLAGS.auxiliary_odom:
      if len(prev_action)==0 and FLAGS.feed_previous_action: raise IOError('previous action was not provided to model.forward.') 
      tensors.append(self.pred_odom)
      feed_dict[self.prev_action] = prev_action
    if len(targets) != 0 and not FLAGS.lstm and (not FLAGS.rl or FLAGS.auxiliary_ctr): 
      tensors.extend([self.total_loss, self.loss])
      feed_dict[self.targets] = targets
    if len(target_depth) != 0 and not FLAGS.lstm:
      if FLAGS.auxiliary_depth: tensors.append(self.depth_loss)
      feed_dict[self.depth_targets] = target_depth
    if len(target_odom) != 0 and not FLAGS.lstm and FLAGS.auxiliary_odom: 
      if len(prev_action)==0 and FLAGS.feed_previous_action: raise IOError('previous action was not provided to model.forward.') 
      tensors.append(self.odom_loss)
      feed_dict[self.odom_targets] = target_odom
      feed_dict[self.prev_action] = prev_action
    
    results = self.sess.run(tensors, feed_dict=feed_dict)
    losses = {}
    aux_results = {}
    state = []
    control = results.pop(0)
    if FLAGS.lstm: 
      state = results.pop(0)
    if auxdepth and FLAGS.auxiliary_depth: aux_results['depth']=results.pop(0)
    if auxodom and FLAGS.auxiliary_odom: aux_results['odom']=results.pop(0)
    if len(targets) != 0 and not FLAGS.lstm and (not FLAGS.rl or FLAGS.auxiliary_ctr):
      losses['t']=results.pop(0) # total loss
      losses['c']=results.pop(0) # control loss
    if len(target_depth) != 0 and not FLAGS.lstm:
      if FLAGS.auxiliary_depth: losses['d']=results.pop(0) # depth loss
    if len(target_odom) != 0 and not FLAGS.lstm and FLAGS.auxiliary_odom:
      losses['o']=results.pop(0) # odometry loss
    return control, state, losses, aux_results

  def backward(self, inputs, initial_state=[], targets=[], depth_targets=[], odom_targets=[], prev_action=[]):
    '''run forward pass and return action prediction
    '''
    tensors = [self.outputs, self.train_op, self.total_loss]
    feed_dict = {self.inputs: inputs}
    if not FLAGS.rl or FLAGS.auxiliary_ctr:
      tensors.append(self.loss)
      feed_dict[self.targets]=targets
    if FLAGS.lstm:
      assert len(initial_state)!=0
      # print 'initial_state: ',initial_state.shape
      feed_dict[self.initial_state]=initial_state
    if (FLAGS.auxiliary_depth or FLAGS.rl) and len(depth_targets)!=0:
      feed_dict[self.depth_targets] = depth_targets
    if FLAGS.auxiliary_depth: tensors.append(self.depth_loss)
    if FLAGS.rl: tensors.append(self.cost_to_go)
    if FLAGS.auxiliary_odom and len(odom_targets)!=0:
      if FLAGS.feed_previous_action and len(prev_action)==0: 
        raise IOError('previous action was not provided to model.backward.') 
      tensors.append(self.odom_loss)
      feed_dict[self.odom_targets] = odom_targets 
      feed_dict[self.prev_action] = prev_action 
    
    results = self.sess.run(tensors, feed_dict=feed_dict)
    
    control = results.pop(0) # control always first
    _ = results.pop(0) # and train_op second
    losses = {'t':results.pop(0)} # total loss
    if not FLAGS.rl or FLAGS.auxiliary_ctr: losses['c']=results.pop(0) # control loss
    if FLAGS.auxiliary_depth and len(depth_targets)!=0: losses['d']=results.pop(0)
    if FLAGS.rl and len(depth_targets)!=0: losses['q']=results.pop(0)
    if FLAGS.auxiliary_odom and len(odom_targets)!=0: losses['o']=results.pop(0)
    
    return control, losses
  
  def get_endpoint_activations(self, inputs):
    '''Run forward through the network for this batch and return all activations
    of all intermediate endpoints
    '''
    tensors = [ self.endpoints[ep] for ep in self.endpoints]
    activations = self.sess.run(tensors, feed_dict={self.inputs:inputs})
    return [ a.reshape(-1,1) for a in activations]

  def fig2buf(self, fig):
    """
    Convert a plt fig to a numpy buffer
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = (h, w, 4)
    
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis = 2 )
    # buf = buf[0::1,0::1] #slice to make image 4x smaller and use only the R channel of RGBA
    buf = buf[0::1,0::1, 0:3] #slice to make image 4x smaller and use only the R channel of RGBA
    #buf = np.resize(buf,(500,500,1))
    return buf

  def plot_with_labels(self, low_d_weights, targets):
    assert low_d_weights.shape[0] >= len(targets), "More targets than weights"
    fig = plt.figure(figsize=(5,5))  #in inches
    for i, label in enumerate(targets):
        x, y = low_d_weights[i,:]
        plt.scatter(x, y,s=50)
        plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom',
                 size='medium')
    buf = self.fig2buf(fig)
    plt.clf()
    plt.close()
    return buf
    # plt.show()
    # plt.savefig(filename)

  def plot_activations(self, inputs, targets=None):
    activation_images = []
    tensors = []
    if FLAGS.network == 'inception':
      endpoint_names = ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3', 'Conv2d_3b_1x1']
      for endpoint in endpoint_names:
        tensors.append(self.endpoints[endpoint])
      units = self.sess.run(tensors, feed_dict={self.inputs:inputs[0:1]})
      for j, unit in enumerate(units):
        filters = unit.shape[3]
        fig = plt.figure(1, figsize=(15,15))
        fig.suptitle(endpoint_names[j], fontsize=40)
        n_columns = 6
        n_rows = math.ceil(filters / n_columns) + 1
        for i in range(filters):
            plt.subplot(n_rows, n_columns, i+1)
            #plt.title('Filter ' + str(i), fontdict={'fontsize':10})
            plt.imshow(unit[0,:,:,i], interpolation="nearest", cmap="gray")
            plt.axis('off')
        #plt.show()
        buf=self.fig2buf(fig)
        plt.clf()
        plt.close()
        #plt.matshow(buf[:,:,0], fignum=100, cmap=plt.cm.gray)
        #plt.axis('off')
        #plt.show()
        #import pdb; pdb.set_trace()
        activation_images.append(buf)
    # Plot using t-SNE
    elif FLAGS.network == 'depth':
      print('shape inputs: {}'.format(inputs.shape))
      activations = self.sess.run(self.endpoints['Conv_4'], feed_dict={self.inputs:inputs})
      tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
      plot_only = 6
      print('shape of activations: {}'.format(activations.shape))
      low_d_weights = tsne.fit_transform(activations)
      activation_images.append(self.plot_with_labels(low_d_weights, targets))
    else:
      raise IOError('MODEL: cant plot activations for this network')
    activation_images = np.asarray(activation_images)
    return activation_images

  def plot_depth(self, inputs, depth_targets):
    '''plot depth predictions and return np array as floating image'''
    if not FLAGS.auxiliary_depth: raise IOError('can t plot depth predictions when auxiliary depth is False.')
    depths = self.sess.run(self.pred_depth, feed_dict={self.inputs: inputs})
    n=3
    fig = plt.figure(1, figsize=(5,5))
    fig.suptitle('depth predictions', fontsize=20)
    for i in range(n):
      plt.axis('off') 
      plt.subplot(n, 3, 1+3*i)
      if FLAGS.n_fc: plt.imshow(inputs[i][:,:,0+3*(FLAGS.n_frames-1):]*1/255.)
      else : plt.imshow(inputs[i][:,:,0:3]*1/255.)
      plt.axis('off') 
      plt.subplot(n, 3, 2+3*i)
      plt.imshow(depths[i]*1/5.)
      plt.axis('off') 
      plt.subplot(n, 3, 3+3*i)
      plt.imshow(depth_targets[i]*1/5.)
      plt.axis('off')
    buf=self.fig2buf(fig)
    # plt.show()
    # import pdb; pdb.set_trace()
    return np.asarray(buf).reshape((1,500,500,3))

  def save(self, logfolder):
    '''save a checkpoint'''
    #self.saver.save(self.sess, logfolder+'/my-model', global_step=run)
    self.saver.save(self.sess, logfolder+'/my-model', global_step=tf.train.global_step(self.sess, self.global_step))
    #self.saver.save(self.sess, logfolder+'/my-model')

  def add_summary_var(self, name):
    var_name = tf.Variable(0., name=name)
    self.summary_vars[name]=var_name
    self.summary_ops[name] = tf.summary.scalar(name, var_name)
    
  def build_summaries(self): 
    self.summary_vars = {}
    self.summary_ops = {}
    # self.summary_vars = []
    self.add_summary_var("Distance_current")
    self.add_summary_var("Distance_current_forest_real")
    self.add_summary_var("Distance_current_sandbox")
    self.add_summary_var("Distance_current_forest")
    self.add_summary_var("Distance_current_canyon")
    self.add_summary_var("Distance_current_esat_corridor_v1")
    self.add_summary_var("Distance_current_esat_corridor_v2")
    self.add_summary_var("Distance_furthest")
    self.add_summary_var("Distance_furthest_forest_real")
    self.add_summary_var("Distance_furthest_sandbox")
    self.add_summary_var("Distance_furthest_forest")
    self.add_summary_var("Distance_furthest_canyon")
    self.add_summary_var("Distance_furthest_esat_corridor_v1")
    self.add_summary_var("Distance_furthest_esat_corridor_v2")
    self.add_summary_var("Distance_average")
    self.add_summary_var("Distance_average_eva")
    self.add_summary_var("Loss_total")
    self.add_summary_var("Loss_control")
    self.add_summary_var("Loss_depth")
    self.add_summary_var("Loss_odom")
    self.add_summary_var("Loss_q")
    self.add_summary_var("Loss_total_eva")
    self.add_summary_var("Loss_control_eva")
    self.add_summary_var("Loss_depth_eva")
    self.add_summary_var("Loss_odom_eva")
    self.add_summary_var("Loss_q_eva")
    self.add_summary_var("odom_errx")
    self.add_summary_var("odom_erry")
    self.add_summary_var("odom_errz")
    self.add_summary_var("odom_erryaw")

    if FLAGS.plot_activations:
      name = "conv_activations"
      act_images = tf.placeholder(tf.float32, [None, 500, 500, 3])
      self.summary_vars[name]=act_images
      self.summary_ops[name]=tf.summary.image(name, act_images, max_outputs=4)
      
    if FLAGS.auxiliary_depth and FLAGS.plot_depth:
      name="depth_predictions"
      dep_images = tf.placeholder(tf.float32, [None, 500, 500, 3])
      self.summary_vars[name]=dep_images
      self.summary_ops[name]=tf.summary.image(name, dep_images, max_outputs=4)
    
    activations={}
    if FLAGS.plot_histograms:
      for ep in self.endpoints: # add activations to summary
        name='activations_{}'.format(ep)
        activations[ep]=tf.placeholder(tf.float32,[None, 1])
        self.summary_vars[name]=activations[ep]
        self.summary_ops[name]=tf.summary.histogram(name, activations[ep])
        
    # self.summary_ops = tf.summary.merge_all()

  def summarize(self, sumvars):
    '''write summary vars with ops'''
    if self.writer:
      feed_dict={self.summary_vars[key]:sumvars[key] for key in sumvars.keys()}
      # feed_dict={self.summary_vars[i]:sumvars[i] for i in range(len(sumvars))}
      sum_op = tf.summary.merge([self.summary_ops[key] for key in sumvars.keys()])
      summary_str = self.sess.run(sum_op, feed_dict=feed_dict)
      self.writer.add_summary(summary_str,  tf.train.global_step(self.sess, self.global_step))
      self.writer.flush()
    
  
  
