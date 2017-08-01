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
tf.app.flags.DEFINE_string("network", 'depth', "Define the type of network: inception / fc_control / depth / mobile.")
tf.app.flags.DEFINE_boolean("auxiliary_depth", False, "Specify whether the horizontal line of depth is predicted as auxiliary task in the feature.")
tf.app.flags.DEFINE_boolean("plot_depth", False, "Specify whether the depth predictions is saved as images.")
tf.app.flags.DEFINE_boolean("n_fc", False, "In case of True, prelogit features are concatenated before feeding to the fully connected layers.")

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


# Use the main method for starting the training procedure and closing it in the end.
def main(_):
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
  
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
  config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  model = Model(sess, state_dim, action_dim, bound=FLAGS.action_bound)
  writer = tf.summary.FileWriter(summary_dir+FLAGS.log_tag, sess.graph)
  model.writer = writer
    
  #model = None
  if FLAGS.launch_ros:
    rosinterface.launch()
  rosnode = rosinterface.PilotNode(model, summary_dir+FLAGS.log_tag)
    
  #def kill_callback(msg):
    #global rosnode
    #print("MAIN: dereferenced rosnode")
    #rosnode = None
    #time.sleep(6)
    ##gzservercount=0
    ##while gzservercount == 0:
      ##gzservercount = os.popen("ps -Af").read().count('gzserver')
      ##time.sleep(0.1)
    #print ("MAIN: gzserver is relaunched!")
    #rosnode = rosinterface.PilotNode(model, summary_dir+FLAGS.log_tag)
  
  #rospy.Subscriber('/roskill', Empty, kill_callback)

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
