
import tensorflow as tf
import os
# import tensorflow.contrib.losses as losses
import tensorflow.contrib.slim as slim
#from tensorflow.contrib.slim.nets import inception
import inception
import fc_control
from tensorflow.contrib.slim import model_analyzer as ma
from tensorflow.python.ops import variables as tf_variables

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

import numpy as np
import math


import depth_estim
import mobile_net

FLAGS = tf.app.flags.FLAGS

# Weight decay of inception network
tf.app.flags.DEFINE_float("weight_decay", 0.00001, "Weight decay of inception network")
# Std of uniform initialization
tf.app.flags.DEFINE_float("init_scale", 0.0027, "Std of uniform initialization")
# Base learning rate
tf.app.flags.DEFINE_float("learning_rate", 0.01, "Start learning rate.")
tf.app.flags.DEFINE_float("depth_weight", 0.01, "Define the weight applied to the depth values in the loss relative to the control loss.")
# Specify where the Model, trained on ImageNet, was saved.
tf.app.flags.DEFINE_string("model_path", 'depth_net_checkpoint', "Specify where the Model, trained on ImageNet, was saved: PATH/TO/vgg_16.ckpt, inception_v3.ckpt or ")
# tf.app.flags.DEFINE_string("model_path", '/users/visics/kkelchte/tensorflow/models', "Specify where the Model, trained on ImageNet, was saved: PATH/TO/vgg_16.ckpt, inception_v3.ckpt or ")
# Define the initializer
#tf.app.flags.DEFINE_string("initializer", 'xavier', "Define the initializer: xavier or uniform [-0.03, 0.03]")
tf.app.flags.DEFINE_string("checkpoint_path", '2017-04-23_1016_esat_cont_depth0420_1514', "Specify the directory of the checkpoint of the earlier trained model.")
tf.app.flags.DEFINE_boolean("continue_training", True, "Specify whether the training continues from a checkpoint or from a imagenet-pretrained model.")
tf.app.flags.DEFINE_boolean("grad_mul", False, "Specify whether the weights of the final tanh activation should be learned faster.")
tf.app.flags.DEFINE_boolean("freeze", False, "Specify whether feature extracting network should be frozen and only the logit scope should be trained.")
tf.app.flags.DEFINE_integer("exclude_from_layer", 8, "In case of training from model (not continue_training), specify up untill which layer the weights are loaded: 5-6-7-8. Default 8: only leave out the logits and auxlogits.")
tf.app.flags.DEFINE_boolean("plot_activations", False, "Specify whether the activations are weighted.")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.9, "Specify the probability of dropout to keep the activation.")
tf.app.flags.DEFINE_integer("clip_grad", 0, "Specify the max gradient norm: default 0, recommended 4.")
tf.app.flags.DEFINE_string("optimizer", 'adadelta', "Specify optimizer, options: adam, adadelta. (!) Adam seems to be unstable as it lead to inf loss.")
tf.app.flags.DEFINE_boolean("plot_histograms", False, "Specify whether to plot histograms of the weights.")

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
    self.lr = FLAGS.learning_rate 
    
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
    
    if FLAGS.network == 'fc_control' and not FLAGS.continue_training:
      init_assign_op = None
    else:
      # get latest folder out of training directory if there is no checkpoint file
      if not os.path.isfile(checkpoint_path+'/checkpoint'):
        checkpoint_path = checkpoint_path+'/'+[mpath for mpath in sorted(os.listdir(checkpoint_path)) if os.path.isdir(checkpoint_path+'/'+mpath) and not mpath[-3:]=='val' and os.path.isfile(checkpoint_path+'/'+mpath+'/checkpoint')][-1]
      print('checkpoint: {}'.format(checkpoint_path))
      init_assign_op, init_feed_dict = slim.assign_from_checkpoint(tf.train.latest_checkpoint(checkpoint_path), variables_to_restore)
  
    # create saver for checkpoints
    self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, max_to_keep=5)
    
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
          self.outputs, self.endpoints, self.auxlogits = inception.inception_v3(self.inputs, num_classes=self.output_size, 
            is_training=(not FLAGS.evaluate), dropout_keep_prob=FLAGS.dropout_keep_prob)  
      elif FLAGS.network=='mobile':
        with slim.arg_scope(mobile_net.mobilenet_v1_arg_scope(weight_decay=FLAGS.weight_decay,
                             stddev=FLAGS.init_scale)):
          #Define model with SLIM, second returned value are endpoints to get activations of certain nodes
          self.outputs, self.endpoints = mobile_net.mobilenet_v1(self.inputs, num_classes=self.output_size, 
            is_training=(not FLAGS.evaluate), dropout_keep_prob=FLAGS.dropout_keep_prob)

          self.auxlogits = self.endpoints['aux_fully_connected_1']
          self.controls, _ = mobile_net.mobilenet_v1(self.inputs, num_classes=self.output_size, is_training=False, reuse = True)
          self.pred_depth = _['aux_fully_connected_1']

      elif FLAGS.network == 'fc_control': #in case of fc_control
        with slim.arg_scope(fc_control.fc_control_v1_arg_scope(weight_decay=FLAGS.weight_decay,
                            stddev=FLAGS.init_scale)): 
          self.outputs, self.endpoints = fc_control.fc_control_v1(self.inputs, num_classes=self.output_size, 
            is_training=(not FLAGS.evaluate), dropout_keep_prob=FLAGS.dropout_keep_prob)
          self.controls, _ = fc_control.fc_control_v1(self.inputs, num_classes=self.output_size, 
            is_training=False, dropout_keep_prob=FLAGS.dropout_keep_prob, reuse=True)
          if(self.bound!=1 or self.bound!=0):
            self.outputs = tf.multiply(self.outputs, self.bound) # Scale output to -bound to bound
      elif FLAGS.network=='depth':
        with slim.arg_scope(depth_estim.arg_scope(weight_decay=FLAGS.weight_decay, stddev=FLAGS.init_scale)):
          # Define model with SLIM, second returned value are endpoints to get activations of certain nodes
          self.outputs, self.endpoints = depth_estim.depth_estim_v1(self.inputs, num_classes=self.output_size, is_training=True)
          self.auxlogits = self.endpoints['fully_connected_1']
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

  def define_loss(self):
    '''tensor for calculating the loss
    '''
    with tf.device(self.device):
      self.targets = tf.placeholder(tf.float32, [None, self.output_size])
      # self.loss = losses.mean_squared_error(tf.clip_by_value(self.outputs,1e-10,1.0), self.targets)
      self.loss = tf.losses.mean_squared_error(self.outputs, self.targets)
      if FLAGS.auxiliary_depth:
        # self.depth_targets = tf.placeholder(tf.float32, [None,1,1,64])
        self.depth_targets = tf.placeholder(tf.float32, [None,55,74])
        self.weights = FLAGS.depth_weight*tf.cast(tf.greater(self.depth_targets, 0), tf.float32)
        # self.depth_loss = tf.losses.mean_squared_error(tf.clip_by_value(self.auxlogits,1e-10,1.0), tf.clip_by_value(self.depth_targets,1e-10,1.0), weights=self.weights)
        
        self.depth_loss = tf.losses.mean_squared_error(self.auxlogits,self.depth_targets,weights=self.weights)
        # self.depth_loss = losses.mean_squared_error(self.auxlogits, self.depth_targets, weights=0.0001)
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
      else:
        raise IOError('Model: Unknown optimizer.')
      # Create the train_op and scale the gradients by providing a map from variable
      # name (or variable) to a scaling coefficient:
      if FLAGS.grad_mul:
        gradient_multipliers = {
          'InceptionV3/Logits/final_tanh/weights/read:0': 10,
          'InceptionV3/Logits/final_tanh/biases/read:0': 10,
        }
      else:
        gradient_multipliers = {}
      if FLAGS.freeze:
        global_variables = [v for v in tf.global_variables() if (v.name.find('Adadelta')==-1 and v.name.find('BatchNorm')==-1)]
        control_variables = [v for v in global_variables if v.name.find('control')!=-1]   # changed logits to control
        print('Only training control variables: ',[v.name for v in control_variables])      
        self.train_op = slim.learning.create_train_op(self.total_loss, self.optimizer, global_step=self.global_step, variables_to_train=control_variables, clip_gradient_norm=FLAGS.clip_grad)
      else:
        self.train_op = slim.learning.create_train_op(self.total_loss, self.optimizer, global_step=self.global_step, gradient_multipliers=gradient_multipliers, clip_gradient_norm=FLAGS.clip_grad)
      # self.train_op = slim.learning.create_train_op(self.total_loss, self.optimizer, gradient_multipliers=gradient_multipliers, global_step=self.global_step)
        
  def forward(self, inputs, aux=False, targets=[], depth_targets=[]):
    '''run forward pass and return action prediction
    '''
    tensors = [self.controls]
    feed_dict={self.inputs: inputs}
    if aux: tensors.append(self.pred_depth)
    if len(targets) != 0: 
      tensors.extend([self.total_loss, self.loss])
      feed_dict[self.targets] = targets
    if len(depth_targets) != 0: 
      tensors.append(self.depth_loss)
      feed_dict[self.depth_targets] = depth_targets
    results = self.sess.run(tensors, feed_dict=feed_dict)
    losses = []
    control = results.pop(0)
    if aux: auxdepth = results.pop(0)
    if len(targets)!=0:
      losses.append(results.pop(0)) # total loss
      losses.append(results.pop(0)) # control loss
    if len(depth_targets) != 0:
      losses.append(results.pop(0)) # depth loss
    if aux:
      if len(losses)==0: return control, auxdepth
      else: return control, losses, auxdepth
    else: return control, losses

  def backward(self, inputs, targets, depth_targets=[]):
    '''run forward pass and return action prediction
    '''
    tensors = [self.outputs, self.train_op, self.total_loss, self.loss]
    feed_dict = {self.inputs: inputs, self.targets: targets}
    if FLAGS.auxiliary_depth and len(depth_targets)!=0: 
      tensors.append(self.depth_loss)
      feed_dict[self.depth_targets] = depth_targets
    results = self.sess.run(tensors, feed_dict=feed_dict)
    control = results[0] # control always first, and train_op second
    losses = results[2:] # rest is losses

    # import pdb; pdb.set_trace()
    # plt.subplot(1,2, 1)
    # plt.imshow(depth_targets[0])
    # plt.subplot(1,2, 2)
    # plt.imshow(weights[0])
    # plt.show()
    # import pdb; pdb.set_trace()
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
    control, depths = self.forward(inputs, aux=True)
    n=3
    fig = plt.figure(1, figsize=(5,5))
    fig.suptitle('depth predictions', fontsize=20)
    for i in range(n):
      plt.axis('off') 
      plt.subplot(n, 3, 1+3*i)
      plt.imshow(inputs[i]*1/255.)
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
    var_name = tf.Variable(0.)
    tf.summary.scalar(name, var_name)
    self.summary_vars.append(var_name)
  
  def build_summaries(self): 
    self.summary_vars = []
    self.add_summary_var("Episode_loss")
    self.add_summary_var("Distance")
    self.add_summary_var("Loss_total")
    self.add_summary_var("Loss_control")
    self.add_summary_var("Loss_depth")
    self.add_summary_var("Furthest_point")

    if FLAGS.plot_activations:
      act_images = tf.placeholder(tf.float32, [None, 500, 500, 3])
      tf.summary.image("conv_activations", act_images, max_outputs=4)
      self.summary_vars.append(act_images)
    
    if FLAGS.auxiliary_depth and FLAGS.plot_depth:
      dep_images = tf.placeholder(tf.float32, [None, 500, 500, 3])
      tf.summary.image("depth_predictions", dep_images, max_outputs=4)
      self.summary_vars.append(dep_images)

    activations={}
    if FLAGS.plot_histograms:
      for ep in self.endpoints: # add activations to summary
        activations[ep]=tf.placeholder(tf.float32,[None, 1])
        tf.summary.histogram('activations_{}'.format(ep), activations[ep])
        self.summary_vars.append(activations[ep])

    self.summary_ops = tf.summary.merge_all()

  def summarize(self, sumvars):
    '''write summary vars with ops'''
    if self.writer:
      feed_dict={self.summary_vars[i]:sumvars[i] for i in range(len(sumvars))}
      summary_str = self.sess.run(self.summary_ops, feed_dict=feed_dict)
      self.writer.add_summary(summary_str,  tf.train.global_step(self.sess, self.global_step))
      self.writer.flush()
    
  
  
