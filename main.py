""" 
Inception trained in simulation supervised fashion
Author: Klaas Kelchtermans (based on code of Patrick Emami)
"""
#from lxml import etree as ET
import xml.etree.cElementTree as ET
import tensorflow as tf
import tensorflow.contrib.losses as losses
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim import model_analyzer as ma
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops import random_ops
import rospy
from std_msgs.msg import Empty
import numpy as np
from model import Model
import rosinterface
import inception
import fc_control
import nfc_control
import mobile_net
import depth_estim
import sys, os, os.path
import subprocess
import shutil
import time
import signal

# Block all the ugly printing...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


FLAGS = tf.app.flags.FLAGS

# ==========================
#   Training Parameters
# ==========================
# Soft target update param
tf.app.flags.DEFINE_float("tau", 0.001, "Update target networks in a soft manner.")

# ===========================
#   Utility Parameters
# ===========================
# Print output of ros verbose or not
tf.app.flags.DEFINE_boolean("load_config", False, "Load flags from the configuration file found in the checkpoint path.")
tf.app.flags.DEFINE_boolean("verbose", True, "Print output of ros verbose or not.")
# Directory for storing tensorboard summary results
tf.app.flags.DEFINE_string("summary_dir", 'tensorflow/log/', "Choose the directory to which tensorflow should save the summaries.")
# tf.app.flags.DEFINE_string("summary_dir", '/esat/qayd/kkelchte/tensorflow/online_log/', "Choose the directory to which tensorflow should save the summaries.")
# Add log_tag to overcome overwriting of other log files
tf.app.flags.DEFINE_string("log_tag", 'testing', "Add log_tag to overcome overwriting of other log files.")
# Choose to run on gpu or cpu
tf.app.flags.DEFINE_string("device", '/gpu:0', "Choose to run on gpu or cpu: /cpu:0 or /gpu:0")
# Set the random seed to get similar examples
tf.app.flags.DEFINE_integer("random_seed", 123, "Set the random seed to get similar examples.")
# Overwrite existing logfolder
tf.app.flags.DEFINE_boolean("owr", True, "Overwrite existing logfolder when it is not testing.")
tf.app.flags.DEFINE_float("action_bound", 1.0, "Define between what bounds the actions can go. Default: [-1:1].")
tf.app.flags.DEFINE_boolean("real", False, "Define settings in case of interacting with the real (bebop) drone.")
tf.app.flags.DEFINE_boolean("launch_ros", False, "Launch ros with simulation_supervised.launch.")
tf.app.flags.DEFINE_boolean("evaluate", False, "Just evaluate the network without training.")
tf.app.flags.DEFINE_string("network", 'mobile_small', "Define the type of network: inception / fc_control / depth / mobile / mobile_small.")
tf.app.flags.DEFINE_boolean("auxiliary_depth", True, "Specify whether a depth map is predicted.")
tf.app.flags.DEFINE_boolean("auxiliary_ctr", False, "Specify whether control should be predicted besides RL.")
tf.app.flags.DEFINE_boolean("auxiliary_odom", True, "Specify whether the odometry or change in x,y,z,Y is predicted.")
tf.app.flags.DEFINE_boolean("plot_depth", False, "Specify whether the depth predictions is saved as images.")
tf.app.flags.DEFINE_boolean("lstm", False, "In case of True, cnn-features are fed into LSTM control layers.")
tf.app.flags.DEFINE_boolean("n_fc", True, "In case of True, prelogit features are concatenated before feeding to the fully connected layers.")
tf.app.flags.DEFINE_integer("n_frames", 3, "Specify the amount of frames concatenated in case of n_fc.")
tf.app.flags.DEFINE_integer("batch_size", 16, "Define the size of minibatches.")
tf.app.flags.DEFINE_integer("num_steps", 8, "Define the number of steps the LSTM layers are unrolled.")
tf.app.flags.DEFINE_integer("lstm_hiddensize", 100, "Define the number of hidden units in the LSTM control layer.")

tf.app.flags.DEFINE_boolean("rl", False, "In case of rl, use reinforcement learning to weight the gradients with a cost-to-go estimated from current depth.")

# ===========================
#   Save settings
# ===========================
def save_config(logfolder, file_name = "configuration"):
  """
  save all the FLAG values in a config file / xml file
  """
  print("Save configuration to: ", logfolder)
  root = ET.Element("conf")
  flg = ET.SubElement(root, "flags")
  
  flags_dict = FLAGS.__dict__['__flags']
  for f in flags_dict:
    #print f, flags_dict[f]
    ET.SubElement(flg, f, name=f).text = str(flags_dict[f])
  tree = ET.ElementTree(root)
  tree.write(os.path.join(logfolder,file_name+".xml"), encoding="us-ascii", xml_declaration=True, method="xml")
# ===========================
#   Load settings
# ===========================
def load_config(modelfolder, file_name = "configuration"):
  """
  save all the FLAG values in a config file / xml file
  """
  print("Load configuration from: ", modelfolder)
  tree = ET.parse(os.path.join(modelfolder,file_name+".xml"))
  boollist=['concatenate_depth','concatenate_odom','lstm','auxiliary_odom',
  'n_fc','feed_previous_action','auxiliary_depth','depth_input']
  intlist=['odom_hidden_units','n_frames','lstm_hiddensize']
  floatlist=[]
  stringlist=['network']
  for child in tree.getroot().find('flags'):
    try :
      if child.attrib['name'] in boollist:
        FLAGS.__setattr__(child.attrib['name'], child.text=='True')
        # print 'set:', child.attrib['name'], child.text=='True'
      elif child.attrib['name'] in intlist:
        FLAGS.__setattr__(child.attrib['name'], int(child.text))
        # print 'set:', child.attrib['name'], int(child.text)
      elif child.attrib['name'] in floatlist:
        FLAGS.__setattr__(child.attrib['name'], float(child.text))
        # print 'set:', child.attrib['name'], float(child.text)
      elif child.attrib['name'] in stringlist:
        FLAGS.__setattr__(child.attrib['name'], str(child.text))
        # print 'set:', child.attrib['name'], str(child.text)
    except : 
      print 'couldnt set:', child.attrib['name'], child.text
      pass

# Use the main method for starting the training procedure and closing it in the end.
def main(_):
  if FLAGS.load_config:
    checkpoint_path = FLAGS.checkpoint_path
    if checkpoint_path[0]!='/': checkpoint_path = os.path.join(os.getenv('HOME'),'tensorflow/log',checkpoint_path)
    if not os.path.isfile(checkpoint_path+'/checkpoint'):
      checkpoint_path = checkpoint_path+'/'+[mpath for mpath in sorted(os.listdir(checkpoint_path)) if os.path.isdir(checkpoint_path+'/'+mpath) and os.path.isfile(checkpoint_path+'/'+mpath+'/checkpoint')][-1]
    load_config(checkpoint_path)

  summary_dir = os.path.join(os.getenv('HOME'),FLAGS.summary_dir)
  # summary_dir = FLAGS.summary_dir
  print("summary dir: {}".format(summary_dir))
  #Check log folders and if necessary remove:
  if FLAGS.log_tag == 'testing' or FLAGS.owr:
    if os.path.isdir(summary_dir+FLAGS.log_tag):
      shutil.rmtree(summary_dir+FLAGS.log_tag,ignore_errors=False)
  else :
    if os.path.isdir(summary_dir+FLAGS.log_tag):
      raise NameError( 'Logfolder already exists, overwriting alert: '+ summary_dir+FLAGS.log_tag ) 
  os.mkdir(summary_dir+FLAGS.log_tag)
  save_config(summary_dir+FLAGS.log_tag)
    
  # some startup settings
  np.random.seed(FLAGS.random_seed)
  tf.set_random_seed(FLAGS.random_seed)
    
  #define the size of the network input
  if FLAGS.network == 'inception':
    state_dim = [1, inception.inception_v3.default_image_size, inception.inception_v3.default_image_size, 3]
  elif FLAGS.network == 'fc_control':
    state_dim = [1, fc_control.fc_control_v1.input_size]
  elif FLAGS.network == 'nfc_control':
    state_dim = [1, nfc_control.fc_control_v1.input_size/4]
  elif FLAGS.network =='depth':
    state_dim = depth_estim.depth_estim_v1.input_size
  elif FLAGS.network =='mobile':
    state_dim = [1, mobile_net.mobilenet_v1.default_image_size, mobile_net.mobilenet_v1.default_image_size, 3]  
  elif FLAGS.network =='mobile_small':
    state_dim = [1, mobile_net.mobilenet_v1.default_image_size_small, mobile_net.mobilenet_v1.default_image_size_small, 3]  
  elif FLAGS.network =='mobile_medium':
    state_dim = [1, mobile_net.mobilenet_v1.default_image_size_medium, mobile_net.mobilenet_v1.default_image_size_medium, 3]  
  else:
    raise NameError( 'Network is unknown: ', FLAGS.network)
    
  action_dim = 1 #initially only turn and go straight
  
  print( "Number of State Dimensions:", state_dim)
  print( "Number of Action Dimensions:", action_dim)
  print( "Action bound:", FLAGS.action_bound)
  # import pdb; pdb.set_trace()
  # tf.logging.set_verbosity(tf.logging.DEBUG)
  # inputs=random_ops.random_uniform(state_dim)
  # targets=random_ops.random_uniform((1,action_dim))
  # depth_targets=random_ops.random_uniform((1,1,1,64))
  
  # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
  # config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
  config=tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = False
  sess = tf.Session(config=config)
  model = Model(sess, state_dim, action_dim, bound=FLAGS.action_bound)
  writer = tf.summary.FileWriter(summary_dir+FLAGS.log_tag, sess.graph)
  model.writer = writer
    
  if FLAGS.launch_ros:
    rosinterface.launch()
  rosnode = rosinterface.PilotNode(model, summary_dir+FLAGS.log_tag)
  
# Random input from tensorflow (could be placeholder)
  #for i in range(10):
    #inpt, trgt, dtrgt = sess.run([inputs, targets, depth_targets])
    ##print('input: ', inpt,' trgt: ',trgt,'dtrgt:', dtrgt)
    ##action = model.forward(inpt)
    ##print('fw: output: ', action)
    #res = model.backward(inpt, trgt, dtrgt)
    #model.summarize([0,0,res[1][0],res[1][1],res[1][2]])
  ##import pdb; pdb.set_trace()
    ##print('bw: {0}'.format(res))
    ##action, loss, depth_action, depth_loss, total, dtotal = model.backward(inpt, trgt, dtrgt)
    ##print('bw: output: {0} loss: {1} \n depth output: {2} depth loss: {3} \n total loss {4} total loss depth {5}'.format(action, loss, depth_action, depth_loss, total, dtotal))
    
   
  
  def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    if FLAGS.launch_ros:
      #model.save(summary_dir+FLAGS.log_tag)
      rosinterface.close()
    sess.close()
    print('done.')
    sys.exit(0)
  signal.signal(signal.SIGINT, signal_handler)
  print('------------Press Ctrl+C to end the learning')

  while True:
    try:
      sys.stdout.flush()
      signal.pause()
    except Exception as e:
      print('! EXCEPTION: ',e)
      if FLAGS.launch_ros:
        rosinterface.close()
      sess.close()
      print('done')
      sys.exit(0)
    
if __name__ == '__main__':
  tf.app.run() 
