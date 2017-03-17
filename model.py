import tensorflow as tf
import tensorflow.contrib.losses as losses
import tensorflow.contrib.slim as slim
import inception
import fc_control
from tensorflow.contrib.slim import model_analyzer as ma
from tensorflow.python.ops import variables as tf_variables

import matplotlib.pyplot as plt
import numpy as np
import math

FLAGS = tf.app.flags.FLAGS

# Weight decay of inception network
tf.app.flags.DEFINE_float("weight_decay", 0.00001, "Weight decay of inception network")
# Std of uniform initialization
tf.app.flags.DEFINE_float("init_scale", 0.0027, "Std of uniform initialization")
# Base learning rate
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Start learning rate.")
# Specify where the Model, trained on ImageNet, was saved.
tf.app.flags.DEFINE_string("model_path", '/home/klaas/tensorflow2/models/inception_v3.ckpt', "Specify where the Model, trained on ImageNet, was saved: PATH/TO/vgg_16.ckpt, inception_v3.ckpt or ")
# Define the initializer
#tf.app.flags.DEFINE_string("initializer", 'xavier', "Define the initializer: xavier or uniform [-0.03, 0.03]")
tf.app.flags.DEFINE_string("checkpoint_path", '/home/klaas/tensorflow2/models/2017-02-15_1923_test/', "Specify the directory of the checkpoint of the earlier trained model.")
tf.app.flags.DEFINE_boolean("continue_training", False, "Specify whether the training continues from a checkpoint or from a imagenet-pretrained model.")
tf.app.flags.DEFINE_boolean("grad_mul", False, "Specify whether the weights of the final tanh activation should be learned faster.")
tf.app.flags.DEFINE_integer("exclude_from_layer", 8, "In case of training from model (not continue_training), specify up untill which layer the weights are loaded: 5-6-7-8. Default 8: only leave out the logits and auxlogits.")
tf.app.flags.DEFINE_boolean("save_activations", False, "Specify whether the activations are weighted.")

"""
Build basic NN model
"""
class Model(object):
 
  def __init__(self,  session, input_size, output_size, prefix='model', device='/gpu:0', bound=1, writer=None):
    '''initialize model
    '''
    self.sess = session
    self.output_size = output_size
    self.bound=bound
    self.input_size = input_size
    self.input_size[0] = None
    #print 'input size: ',self.input_size
    self.prefix = prefix
    self.device = device
    self.writer = writer
    self.lr = FLAGS.learning_rate 
    #if FLAGS.initializer == 'xavier':
      #self.initializer=tf.contrib.layers.xavier_initializer()
    #else:
      #self.initializer = tf.random_uniform_initializer(-init_scale, init_scale)
    # need to catch variables to restore before defining the training op
    # because adam variables are not available in check point.
    
    # build network from SLIM model
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.define_network()
    
    if FLAGS.network == 'inception' and not FLAGS.continue_training:
      checkpoint_path = FLAGS.model_path
      list_to_exclude = []
      if FLAGS.exclude_from_layer <= 7:
        list_to_exclude.extend(["InceptionV3/Mixed_7a", "InceptionV3/Mixed_7b", "InceptionV3/Mixed_7c"])
      if FLAGS.exclude_from_layer <= 6:
        list_to_exclude.extend(["InceptionV3/Mixed_6a", "InceptionV3/Mixed_6b", "InceptionV3/Mixed_6c", "InceptionV3/Mixed_6d", "InceptionV3/Mixed_6e"])
      if FLAGS.exclude_from_layer <= 5:
        list_to_exclude.extend(["InceptionV3/Mixed_5a", "InceptionV3/Mixed_5b", "InceptionV3/Mixed_5c", "InceptionV3/Mixed_5d"])
      
      list_to_exclude.extend(["InceptionV3/Logits", "InceptionV3/AuxLogits"])
      #print list_to_exclude
      variables_to_restore = slim.get_variables_to_restore(exclude=list_to_exclude)
      init_assign_op, init_feed_dict = slim.assign_from_checkpoint(checkpoint_path, variables_to_restore)
    if FLAGS.continue_training:
      variables_to_restore = slim.get_variables_to_restore()
      if FLAGS.checkpoint_path[0]!='/':
        checkpoint_path = '/home/klaas/tensorflow2/log/'+FLAGS.checkpoint_path
      else:
        checkpoint_path = FLAGS.checkpoint_path
      init_assign_op, init_feed_dict = slim.assign_from_checkpoint(tf.train.latest_checkpoint(checkpoint_path), variables_to_restore)
      
    # Add the loss function to the graph.
    self.define_loss()
        
    if not FLAGS.evaluate:
      # create saver for checkpoints
      self.saver = tf.train.Saver()
      
      # Define the training op based on the total loss
      self.define_train()
    
    # Define summaries
    self.build_summaries()
    
    init_all=tf_variables.global_variables_initializer()
    self.sess.run([init_all])
    if FLAGS.network == 'inception' or FLAGS.continue_training:
      self.sess.run([init_assign_op], init_feed_dict)
      #self.sess.run([init_all,init_assign_op], init_feed_dict)
      print('Successfully loaded model:',checkpoint_path)
    
  def define_network(self):
    '''build the network and set the tensors
    '''
    with tf.device(self.device):
      self.inputs = tf.placeholder(tf.float32, shape = self.input_size)
      if FLAGS.network == 'inception':
        #inputs = tf.placeholder(tf.float32, shape = [None, 299, 299, 3])
        ### initializer is defined in the arg scope of inception.
        ### need to stick to this arg scope in order to load the inception checkpoint properly...
        ### weights are now initialized uniformly 
        with slim.arg_scope(inception.inception_v3_arg_scope(weight_decay=FLAGS.weight_decay,
                            stddev=FLAGS.init_scale)):
          #Define model with SLIM, second returned value are endpoints to get activations of certain nodes
          self.outputs, self.endpoints, self.auxlogits = inception.inception_v3(self.inputs, num_classes=self.output_size, is_training=True)  
          if(self.bound!=1 or self.bound!=0):
            self.outputs = tf.mul(self.outputs, self.bound) # Scale output to -bound to bound
      else: #in case of fc_control
        with slim.arg_scope(fc_control.fc_control_v1_arg_scope(weight_decay=FLAGS.weight_decay,
                            stddev=FLAGS.init_scale)):
          self.outputs, _ = fc_control.fc_control_v1(self.inputs, num_classes=self.output_size, is_training=True)
          if(self.bound!=1 or self.bound!=0):
            self.outputs = tf.mul(self.outputs, self.bound) # Scale output to -bound to bound
            
  def define_loss(self):
    '''tensor for calculating the loss
    '''
    with tf.device(self.device):
      self.targets = tf.placeholder(tf.float32, [None, self.output_size])
      self.loss = losses.mean_squared_error(self.outputs, self.targets)
      self.total_loss = losses.get_total_loss()
      
  def define_train(self):
    '''applying gradients to the weights from normal loss function
    '''
    with tf.device(self.device):
      # Specify the optimizer and create the train op:  
      #self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr) 
      self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr) 
      # Create the train_op and scale the gradients by providing a map from variable
      # name (or variable) to a scaling coefficient:
      if FLAGS.grad_mul:
        gradient_multipliers = {
          'InceptionV3/Logits/final_tanh/weights/read:0': 10,
          'InceptionV3/Logits/final_tanh/biases/read:0': 10,
        }
      else:
        gradient_multipliers = {}
      self.train_op = slim.learning.create_train_op(self.total_loss, self.optimizer, gradient_multipliers=gradient_multipliers, global_step=self.global_step)
        
  def forward(self, inputs, targets=None):
    '''run forward pass and return action prediction
    '''
    if targets == None:
      return self.sess.run(self.outputs, feed_dict={self.inputs: inputs})
    else:
      return self.sess.run([self.outputs, self.total_loss], feed_dict={self.inputs: inputs, self.targets: targets})
    
  def backward(self, inputs, targets):
    '''run forward pass and return action prediction
    '''
    control, loss, _ = self.sess.run([self.outputs, self.total_loss, self.train_op], feed_dict={self.inputs: inputs, self.targets: targets})
    
    return control, loss
  
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
    buf = buf[0::1,0::1,0:1] #slice to make image 4x smaller and use only the R channel of RGBA
    #buf = np.resize(buf,(500,500,1))
    return buf
  
  def plot_activations(self, inputs):
    activation_images = []
    tensors = []
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
      activation_images.append(buf)
      plt.clf()
      plt.close()
      #plt.matshow(buf[:,:,0], fignum=100, cmap=plt.cm.gray)
      #plt.axis('off')
      #plt.show()
      #import pdb; pdb.set_trace()
    activation_images = np.asarray(activation_images)
    
    return activation_images
  
  def save(self, logfolder):
    '''save a checkpoint'''
    self.saver.save(self.sess, logfolder+'/my-model', global_step=tf.train.global_step(self.sess, self.global_step))
  
  def build_summaries(self): 
    episode_loss = tf.Variable(0.)
    tf.summary.scalar("Loss", episode_loss)
    distance = tf.Variable(0.)
    tf.summary.scalar("Distance", distance)
    batch_loss = tf.Variable(0.)
    tf.summary.scalar("Batch_loss", batch_loss)
    if FLAGS.save_activations:
      act_images = tf.placeholder(tf.float32, [None, 1500, 1500, 1])
      tf.summary.image("conv_activations", act_images, max_outputs=4)
      self.summary_vars = [episode_loss, distance, batch_loss, act_images]
    else:
      self.summary_vars = [episode_loss, distance, batch_loss]
    self.summary_ops = tf.summary.merge_all()

  def summarize(self, sumvars):
    '''write summary vars with ops'''
    if self.writer:
      feed_dict={self.summary_vars[i]:sumvars[i] for i in range(len(sumvars))}
      summary_str = self.sess.run(self.summary_ops, feed_dict=feed_dict)
      self.writer.add_summary(summary_str,  tf.train.global_step(self.sess, self.global_step))
      self.writer.flush()
    
  
  
