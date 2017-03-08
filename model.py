import tensorflow as tf
import tensorflow.contrib.losses as losses
import tensorflow.contrib.slim as slim
#from tensorflow.contrib.slim.nets import inception
import inception
from tensorflow.contrib.slim import model_analyzer as ma
from tensorflow.python.ops import variables as tf_variables

FLAGS = tf.app.flags.FLAGS

# Weight decay of inception network
tf.app.flags.DEFINE_float("weight_decay", 0.00001, "Weight decay of inception network")
# Std of uniform initialization
tf.app.flags.DEFINE_float("init_scale", 0.0027, "Std of uniform initialization")
# Base learning rate
tf.app.flags.DEFINE_float("learning_rate", 0.00001, "Start learning rate.")
# Specify where the Model, trained on ImageNet, was saved.
tf.app.flags.DEFINE_string("model_path", '/home/klaas/tensorflow2/models/inception_v3.ckpt', "Specify where the Model, trained on ImageNet, was saved: PATH/TO/vgg_16.ckpt, inception_v3.ckpt or ")
# Define the initializer
#tf.app.flags.DEFINE_string("initializer", 'xavier', "Define the initializer: xavier or uniform [-0.03, 0.03]")
tf.app.flags.DEFINE_string("checkpoint_path", '/home/klaas/tensorflow2/models/2017-02-15_1923_test/', "Specify the directory of the checkpoint of the earlier trained model.")
tf.app.flags.DEFINE_boolean("continue_training", False, "Specify whether the training continues from a checkpoint or from a imagenet-pretrained model.")
tf.app.flags.DEFINE_boolean("grad_mul", False, "Specify whether the weights of the final tanh activation should be learned faster.")

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
    self.define_network()
    if not FLAGS.continue_training and not FLAGS.evaluate:
      checkpoint_path = FLAGS.model_path
      variables_to_restore = slim.get_variables_to_restore(exclude=["InceptionV3/Logits", "InceptionV3/AuxLogits"])
      init_assign_op, init_feed_dict = slim.assign_from_checkpoint(checkpoint_path, variables_to_restore)
    else:
      variables_to_restore = slim.get_variables_to_restore()
      if FLAGS.checkpoint_path[0]!='/':
        checkpoint_path = '/home/klaas/tensorflow2/log/'+FLAGS.checkpoint_path
      else:
        checkpoint_path = FLAGS.checkpoint_path
      init_assign_op, init_feed_dict = slim.assign_from_checkpoint(tf.train.latest_checkpoint(checkpoint_path), variables_to_restore)
    
    if not FLAGS.evaluate:
      # create saver for checkpoints
      self.saver = tf.train.Saver()
      
      # Add the loss function to the graph.
      self.define_loss()
      
      # Define the training op based on the total loss
      self.define_train()
    
    # Define summaries
    self.build_summaries()
    
    init_all=tf_variables.global_variables_initializer()
    self.sess.run([init_all])
    self.sess.run([init_assign_op], init_feed_dict)
    #self.sess.run([init_all,init_assign_op], init_feed_dict)
    print('Successfully loaded model:',checkpoint_path)
    
  def define_network(self):
    '''build the network and set the tensors
    '''
    with tf.device(self.device):
      self.inputs = tf.placeholder(tf.float32, shape = [None, self.input_size, self.input_size, 3])
      #inputs = tf.placeholder(tf.float32, shape = [None, 299, 299, 3])
      ### initializer is defined in the arg scope of inception.
      ### need to stick to this arg scope in order to load the inception checkpoint properly...
      ### weights are now initialized uniformly 
      with slim.arg_scope(inception.inception_v3_arg_scope(weight_decay=FLAGS.weight_decay,
                           stddev=FLAGS.init_scale)):
        #Define model with SLIM, second returned value are endpoints to get activations of certain nodes
        self.outputs, _ = inception.inception_v3(self.inputs, num_classes=self.output_size, is_training=True)  
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
      self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr) 
      # Create the train_op and scale the gradients by providing a map from variable
      # name (or variable) to a scaling coefficient:
      if FLAGS.grad_mul:
        gradient_multipliers = {
          'InceptionV3/Logits/final_tanh/weights/read:0': 10,
          'InceptionV3/Logits/final_tanh/biases/read:0': 10,
        }
      else:
        gradient_multipliers = {}
      self.train_op = slim.learning.create_train_op(self.total_loss, self.optimizer, gradient_multipliers=gradient_multipliers)
        
  def forward(self, inputs):
    '''run forward pass and return action prediction
    '''
    return self.sess.run(self.outputs, feed_dict={self.inputs: inputs})
  
  def backward(self, inputs, targets):
    '''run forward pass and return action prediction
    '''
    control, loss, _ = self.sess.run([self.outputs, self.total_loss, self.train_op], feed_dict={self.inputs: inputs, self.targets: targets})
    
    return control, loss
  
  def save(self, run, logfolder):
    '''save a checkpoint'''
    self.saver.save(self.sess, logfolder+'/my-model', global_step=run)
  
  def build_summaries(self): 
    episode_loss = tf.Variable(0.)
    tf.summary.scalar("Loss", episode_loss)
    distance = tf.Variable(0.)
    tf.summary.scalar("Distance", distance)
    batch_loss = tf.Variable(0.)
    tf.summary.scalar("Batch_loss", batch_loss)
    self.summary_vars = [episode_loss, distance, batch_loss]
    self.summary_ops = tf.summary.merge_all()

  def summarize(self, run, sumvars):
    '''write summary vars with ops'''
    if self.writer:
      feed_dict={self.summary_vars[i]:sumvars[i] for i in range(len(sumvars))}
      summary_str = self.sess.run(self.summary_ops, feed_dict=feed_dict)
      self.writer.add_summary(summary_str, run)
      self.writer.flush()
    
  
  
