import rospy
import numpy as np
import scipy.misc as sm
import os
import subprocess
from os import path
import rospy
import time
from cv_bridge import CvBridge, CvBridgeError
import cv2
# Instantiate CvBridge
bridge = CvBridge()

from replay_buffer import ReplayBuffer

#from pygazebo import images_stamped_pb2 as gzimages

import tensorflow as tf

from model import Model

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from nav_msgs.msg import Odometry

#from PIL import Image

FLAGS = tf.app.flags.FLAGS
# =================================================
tf.app.flags.DEFINE_string("rundir", 'runs', "Choose location to keep the trajectories.")
#tf.app.flags.DEFINE_string("launchfile", 'simulation_supervised.launch', "Choose launch file. If not starting with / the path is seen as relative to current directory")
tf.app.flags.DEFINE_integer("num_flights", 1000, "the maximum number of tries.")
tf.app.flags.DEFINE_boolean("render", True, "Render the game while it is being learned.")
tf.app.flags.DEFINE_boolean("experience_replay", True, "Accumulate a buffer of experience to learn from.")
tf.app.flags.DEFINE_integer("buffer_size", 100, "Define the number of experiences saved in the buffer.")
tf.app.flags.DEFINE_integer("batch_size", 16, "Define the size of minibatches.")
tf.app.flags.DEFINE_float("mean", 0.2623, "Define the mean of the input data for centering around zero.(sandbox:0.5173,esat:0.2623)")
tf.app.flags.DEFINE_float("std", 0.1565, "Define the standard deviation of the data for normalization.(sandbox:0.3335,esat:0.1565)")
#tf.app.flags.DEFINE_float("gradient_threshold", 0.0001, "The minimum amount of difference between target and estimated control before applying gradients.")
tf.app.flags.DEFINE_boolean("depth_input", False, "Use depth input instead of RGB for training the network.")
tf.app.flags.DEFINE_boolean("reloaded_by_ros", False, "This boolean postpones filling the replay buffer as it is just loaded by ros after a crash. It will keep the target_control None for the three runs.")
tf.app.flags.DEFINE_float("epsilon", 0., "Epsilon is the probability that the control is picked randomly.")
tf.app.flags.DEFINE_float("alpha", 0., "Alpha is the amount of noise in the general y, z and Y direction during training to ensure it visits the whole corridor.")
# =================================================

launch_popen=None

def launch():
  global launch_popen
  """ launch ros, gazebo and this node """
  #start roscore
  #if not FLAGS.real_evaluation:
  if FLAGS.render: render='true'
  else: render='false'
  launchfile='simulation_supervised.launch'
  evaluate='false'
  #subprocess.Popen("/home/klaas/sandbox_ws/src/sandbox/scripts/launch_online.sh", shell=True)
  #else:
    #launchfile='real_world.launch'
    #render='false'
    #evaluate='true'
  #launch_popen=subprocess.Popen("/home/klaas/sandbox_ws/src/sandbox/scripts/launch_online.sh "+launchfile+" "+str(FLAGS.num_flights)+' '+render+' '+evaluate, shell=True)
  launch_popen=subprocess.Popen("/home/klaas/sandbox_ws/src/sandbox/scripts/launch_online.sh "+str(FLAGS.num_flights), shell=True)
    
  #if not FLAGS.real_evaluation:
    # wait for gzserver to be launched:
  gzservercount=0
  while gzservercount == 0:
    #print('gzserver: ',gzservercount)
    gzservercount = os.popen("ps -Af").read().count('gzserver')
    time.sleep(0.1)
  print ("Roscore launched! ", launch_popen.pid)
  
def close():
  """ Kill gzclient, gzserver and roscore"""
  if launch_popen:
    os.popen("kill -9 "+str(launch_popen.pid))
    launch_popen.wait()
  tmp = os.popen("ps -Af").read()
  gzclient_count = tmp.count('gzclient')
  gzserver_count = tmp.count('gzserver')
  roscore_count = tmp.count('roscore')
  rosmaster_count = tmp.count('rosmaster')
  launch_online_count = tmp.count('launch_online')
  print("processes busy: ",' ',str(gzclient_count),' ',str(gzserver_count),' ',str(roscore_count),' ',str(rosmaster_count),' ',str(launch_online_count))
  if gzclient_count > 0:
      os.system("killall -9 gzclient")
  if gzserver_count > 0:
      os.system("killall -9 gzserver")
  if rosmaster_count > 0:
      os.system("killall -9 rosmaster")
  if roscore_count > 0:
      os.system("killall -9 roscore")
  if roscore_count > 0:
      os.system("killall -9 roscore")

  if (gzclient_count or gzserver_count or roscore_count or rosmaster_count >0):
    os.wait()
  
class PilotNode(object):
  
  def __init__(self, model, logfolder):
    # Initialize replay memory
    self.logfolder = logfolder
    self.run=0
    self.maxy=-10
    self.distance=0
    self.last_position=None
    self.model = model
    self.ready=False 
    self.finished=True
    self.target_control = None
    self.target_depth = None
    rospy.init_node('pilot', anonymous=True)
    self.delay_evaluation = 0
    if rospy.has_param('delay_evaluation'):
      self.delay_evaluation=rospy.get_param('delay_evaluation')  
    if FLAGS.real: # in the real world on the bebop drone
      rospy.Subscriber('/bebop/image_raw', Image, self.image_callback)
      self.action_pub = rospy.Publisher('/bebop/cmd_vel', Twist, queue_size=1)
      rospy.Subscriber('/bebop/ready', Empty, self.ready_callback)
      rospy.Subscriber('/bebop/overtake', Empty, self.overtake_callback)
    else: # in simulation
      self.replay_buffer = ReplayBuffer(FLAGS.buffer_size, FLAGS.random_seed)
      self.accumloss = 0
      
      if FLAGS.depth_input or FLAGS.auxiliary_depth:
        rospy.Subscriber('/ardrone/kinect/depth/image_raw', Image, self.depth_image_callback)
      if not FLAGS.depth_input:        
        rospy.Subscriber('/ardrone/kinect/image_raw', Image, self.image_callback)
      self.action_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
      rospy.Subscriber('/ready', Empty, self.ready_callback)
      rospy.Subscriber('/finished', Empty, self.finished_callback)
      #rospy.Subscriber('/finished', Empty, self.overtake_callback)
      #rospy.Subscriber('/ardrone/overtake', Empty, self.overtake_callback)
      rospy.Subscriber('/ground_truth/state', Odometry, self.gt_callback)
      rospy.Subscriber('/supervised_vel', Twist, self.supervised_callback)
    #if FLAGS.evaluate and FLAGS.save_activations: 
      #raise Exception('Cant evaluate and save activations in current implementation.')
      
  def overtake_callback(self, data):
    if self.ready:
      print('Neural control overtaken.')
      self.finished = True
      self.ready = False
      
  def ready_callback(self,msg):
    if not self.ready and self.finished:
      print('Neural control activated.')
      self.ready = True
      self.start_time = rospy.get_time()
      self.finished = False
    
  def gt_callback(self, data):
    if not self.ready: return
    current_pos=[data.pose.pose.position.x,
                    data.pose.pose.position.y,
                    data.pose.pose.position.z]
    #if self.last_position:
      #self.distance += np.sqrt((self.last_position[0]-current_pos[0])**2+(self.last_position[1]-current_pos[1])**2)
    self.distance=max([self.distance, np.sqrt(current_pos[0]**2+current_pos[1]**2)])
    self.last_position=current_pos
    #self.maxy=max([self.last_position[1], self.maxy])
    
    #if self.ready and not self.finished:
      #time.sleep(0.5)
    #print(self.distance)
  
  def image_callback(self, data):
    if not self.ready or self.finished or (rospy.get_time()-self.start_time) < self.delay_evaluation: return
    try:
      # Convert your ROS Image message to OpenCV2
      cv2_img = bridge.imgmsg_to_cv2(data, 'bgr8') 
    except CvBridgeError as e:
      print(e)
    else:
      #print('received image')
      # 360*640*3
      #(rows,cols,channels) = cv2_img.shape
      #im = sm.imresize(cv2_img,(self.model.input_size, self.model.input_size, 3),'nearest')
      im = sm.imresize(cv2_img,(299, 299, 3),'nearest')
      im = im*1/255.
      # Basic preprocessing: center + make 1 standard deviation
      im -= FLAGS.mean
      im = im*1/FLAGS.std
      self.process_input(im)

  def depth_image_callback(self, data):
    if not self.ready or self.finished or (rospy.get_time()-self.start_time) < self.delay_evaluation: return
    try:
      # Convert your ROS Image message to OpenCV2
      im = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')#gets float of 32FC1 depth image
    except CvBridgeError as e:
      print(e)
    else:
      # crop image to get 1 line in the middle
      row = int(im.shape[0]/2.)
      step_colums = int(im.shape[1]/64.)
      arr = im[row, ::step_colums]
      arr_clean = [e if not np.isnan(e) else 5 for e in arr]
      arr_clean = np.array(arr_clean)
      arr_clean = arr_clean*1/5.-0.5
      #print(arr_clean)
    if FLAGS.depth_input:
      self.process_input(arr_clean)
    if FLAGS.auxiliary_depth:
      self.target_depth = arr_clean #(64,) 
    
  def process_input(self, im):
    if self.target_control: 
      trgt = self.target_control[5]
      trgt_depth = None
      if self.target_depth != None:
        trgt_depth = self.target_depth[:]
      if FLAGS.experience_replay:
        # add the experience to the replay buffer
        # shape target control (1)
        if FLAGS.auxiliary_depth:
          self.replay_buffer.add(im,[trgt],[trgt_depth])
        else:
          self.replay_buffer.add(im,[trgt])
        control = self.model.forward([im])
      if not FLAGS.experience_replay:
        if FLAGS.auxiliary_depth:
          control, losses = self.model.backward([im],[[trgt]], [[[trgt_depth]]])
        else:
          control, losses = self.model.backward([im],[[trgt]])
        print 'Difference: '+str(control[0,0])+' and '+str(trgt)+'='+str(abs(control[0,0]-trgt))
        self.accumloss += losses[0]
    else:
      control = self.model.forward([im])
    if self.last_position and self.ready:
      self.runfile = open(self.logfolder+'/runs.txt', 'a')
      self.runfile.write('{0:05d} {1[0]:0.3f} {1[1]:0.3f} {1[2]:0.3f} \n'.format(self.run, self.last_position))
      self.runfile.close()
    yaw = control[0,0]
    if np.random.binomial(1,FLAGS.epsilon):
      yaw = max(-1,min(1,np.random.normal()))
    msg = Twist()
    msg.linear.x = 1.8
    msg.linear.y = np.random.uniform(-FLAGS.alpha, FLAGS.alpha)
    msg.linear.z = np.random.uniform(-FLAGS.alpha, FLAGS.alpha)
    msg.angular.z = yaw
    self.action_pub.publish(msg)
  
  def supervised_callback(self, data):
    if not self.ready: return
    if FLAGS.reloaded_by_ros and self.run<=3: return
    else:
      self.target_control = [data.linear.x,
        data.linear.y,
        data.linear.z,
        data.angular.x,
        data.angular.y,
        data.angular.z]
      #print(self.target_control)
      
  def finished_callback(self,msg):
    if self.ready and not self.finished:
      print('neural control deactivated.')
      self.ready=False
      self.finished=True
      # Train model from experience replay:
      # Train the model with batchnormalization out of the image callback loop
      activation_images = None
      closs = [] #control loss
      dloss = [] #depth loss
      tloss = [] #total loss
      #tot_batch_loss = []
      if FLAGS.experience_replay and self.replay_buffer.size()>FLAGS.batch_size:
        for b in range(min(int(self.replay_buffer.size()/FLAGS.batch_size), 10)):
          #im_b, target_b = self.replay_buffer.sample_batch(FLAGS.batch_size)
          batch = self.replay_buffer.sample_batch(FLAGS.batch_size)
          #print('time to smaple batch of images: ',time.time()-st)
          if b==0 and FLAGS.save_activations:
            activation_images= self.model.plot_activations(batch[0])
          if FLAGS.evaluate:
            # shape control (16,1)
            controls, loss = self.model.forward(batch[0],batch[1][:,0].reshape(-1,1))
            losses = [loss, 0, 0]
          else:
            if FLAGS.auxiliary_depth:
              controls, losses = self.model.backward(batch[0],batch[1][:].reshape(-1,1),batch[2][:].reshape(-1,1,1,64))
            else:
              controls, losses = self.model.backward(batch[0],batch[1][:].reshape(-1,1))
            if len(losses) == 2: losses.append(0) #in case there is no depth
          tloss.append(losses[0])
          closs.append(losses[1])
          dloss.append(losses[2])
        tloss = np.mean(tloss)
        closs = np.mean(closs)
        dloss = np.mean(dloss)
        #batch_loss = np.mean(tot_batch_loss)
        
      else:
        print('filling buffer or no experience_replay: ', self.replay_buffer.size())
        tloss = 0
        closs = 0
        dloss = 0
      try:
        if FLAGS.save_activations and activation_images!=None:
          sumvar=[self.accumloss, self.distance, tloss, closs, dloss, activation_images]
        else:
          sumvar=[self.accumloss, self.distance, tloss, closs, dloss]
        self.model.summarize(sumvar)
      except Exception as e:
        print('failed to write', e)
        pass
      else:
        print('control finished {0}:[ acc loss: {1:0.3f}, distance: {2:0.3f}, total loss: {3:0.3f}, control loss: {4:0.3f}, depth loss: {5:0.3f}]'.format(self.run, self.accumloss, self.distance, tloss, closs, dloss))
      self.accumloss = 0
      self.maxy = -10
      self.distance = 0
      self.last_position = None
      
      if self.run%20==0 and not FLAGS.evaluate:
        # Save a checkpoint every 100 runs.
        self.model.save(self.logfolder)
      
      self.run+=1 
      # wait for gzserver to be killed
      gzservercount=1
      while gzservercount > 0:
        #print('gzserver: ',gzservercount)
        gzservercount = os.popen("ps -Af").read().count('gzserver')
        time.sleep(0.1)