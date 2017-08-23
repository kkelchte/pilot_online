import rospy
import numpy as np
import scipy.misc as sm
import os, sys
import subprocess
from os import path
import rospy
import time
from cv_bridge import CvBridge, CvBridgeError
import cv2
import re
# Instantiate CvBridge
bridge = CvBridge()

from replay_buffer import ReplayBuffer

# Debug odometry
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#from pygazebo import images_stamped_pb2 as gzimages


import inception
import fc_control
import depth_estim
import copy 

import tensorflow as tf

from model import Model

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from nav_msgs.msg import Odometry

from ou_noise import OUNoise

#from PIL import Image

FLAGS = tf.app.flags.FLAGS
# =================================================
tf.app.flags.DEFINE_string("rundir", 'runs', "Choose location to keep the trajectories.")
#tf.app.flags.DEFINE_string("launchfile", 'simulation_supervised.launch', "Choose launch file. If not starting with / the path is seen as relative to current directory")
tf.app.flags.DEFINE_integer("num_flights", 1000, "the maximum number of tries.")
tf.app.flags.DEFINE_boolean("render", False, "Render the game while it is being learned.")
tf.app.flags.DEFINE_boolean("experience_replay", True, "Accumulate a buffer of experience to learn from.")
tf.app.flags.DEFINE_integer("buffer_size", 2000, "Define the number of experiences saved in the buffer.")
tf.app.flags.DEFINE_float("mean", 0., "Define the mean of the input data for centering around zero.(sandbox:0.5173,esat:0.2623)")
tf.app.flags.DEFINE_float("std", 1., "Define the standard deviation of the data for normalization.(sandbox:0.3335,esat:0.1565)")
#tf.app.flags.DEFINE_float("gradient_threshold", 0.0001, "The minimum amount of difference between target and estimated control before applying gradients.")
tf.app.flags.DEFINE_boolean("depth_input", False, "Use depth input instead of RGB for training the network.")
# tf.app.flags.DEFINE_boolean("reloaded_by_ros", False, "This boolean postpones filling the replay buffer as it is just loaded by ros after a crash. It will keep the target_control None for the three runs.")
# tf.app.flags.DEFINE_float("epsilon", 0.1, "Epsilon is the probability that the control is picked randomly.")
tf.app.flags.DEFINE_string("type_of_noise", 'ou', "Define whether the noise is temporally correlated (ou) or uniformly distributed (uni).")
tf.app.flags.DEFINE_float("sigma_z", 0.01, "sigma_z is the amount of noise in the z direction.")
tf.app.flags.DEFINE_float("sigma_x", 0.01, "sigma_x is the amount of noise in the forward speed.")
tf.app.flags.DEFINE_float("sigma_y", 0.01, "sigma_y is the amount of noise in the y direction.")
tf.app.flags.DEFINE_float("sigma_yaw", 0.1, "sigma_yaw is the amount of noise added to the steering angle.")
tf.app.flags.DEFINE_float("speed", 1.3, "Define the forward speed of the quadrotor.")
tf.app.flags.DEFINE_float("alpha",0.,"Policy mixing: choose with a binomial chance of alpha for the experts policy.")

tf.app.flags.DEFINE_boolean("off_policy",False,"In case the network is off_policy, the control is published on supervised_vel instead of cmd_vel.")
tf.app.flags.DEFINE_boolean("show_depth",True,"Publish the predicted horizontal depth array to topic ./depth_prection so show_depth can visualize this in another node.")
tf.app.flags.DEFINE_boolean("show_odom",True,"Publish the predicted odometry on ./odom_prection so show_odom can visualize the estimated trajectory.")
tf.app.flags.DEFINE_boolean("recovery_cameras",False,"Listen to recovery cameras (left-right 30-60) and add them in replay buffer.")
tf.app.flags.DEFINE_boolean("save_input",False,"Write depth input to file in order to check values later.")

tf.app.flags.DEFINE_float("ou_theta", 0.15, "Epsilon is the probability that the control is picked randomly.")
# tf.app.flags.DEFINE_float("ou_sigma", 0.3, "Alpha is the amount of noise in the general y, z and Y direction during training to ensure it visits the whole corridor.")
# =================================================

launch_popen=None

class PilotNode(object):
  
  def __init__(self, model, logfolder):
    print('initialize pilot node')

    np.random.seed(FLAGS.random_seed)
    tf.set_random_seed(FLAGS.random_seed)
    
    # Initialize replay memory
    self.logfolder = logfolder
    self.world_name = ''
    self.logfile = logfolder+'/tensorflow_log'
    self.run=0
    self.run_eva=0
    self.maxy=-10
    self.speed=FLAGS.speed
    self.accumlosses = {}
    self.current_distance=0
    self.furthest_point=0
    self.average_distance=0
    self.average_distance_eva=0
    self.last_position=[]
    self.model = model
    self.ready=False 
    self.finished=True
    self.target_control = []
    self.target_depth = []
    self.target_odom = []
    self.aux_depth = []
    self.aux_odom = []
    self.odom_error = []
    self.prev_control = [0]
    self.nfc_images =[] #used by n_fc networks for building up concatenated frames
    self.nfc_positions =[] #used by n_fc networks for calculating odometry
    rospy.init_node('pilot', anonymous=True)
    self.exploration_noise = OUNoise(4, 0, FLAGS.ou_theta,1)
    self.state = []

    # self.delay_evaluation = 5 #can't be set by ros because node is started before ros is started...
    if FLAGS.show_depth:
      self.depth_pub = rospy.Publisher('/depth_prediction', numpy_msg(Floats), queue_size=1)
    if FLAGS.show_odom:
      self.odom_pub = rospy.Publisher('/odom_prediction', numpy_msg(Floats), queue_size=1)
    # if FLAGS.off_policy:
    #   self.action_pub = rospy.Publisher('/supervised_vel', Twist, queue_size=1)
    #   if rospy.has_param('control'):
    #     rospy.Subscriber(rospy.get_param('control'), Twist, self.supervised_callback)
    if FLAGS.real or FLAGS.off_policy:
      self.action_pub=rospy.Publisher('/pilot_vel', Twist, queue_size=1)
    else:
      rospy.Subscriber('/supervised_vel', Twist, self.supervised_callback)
      if rospy.has_param('control'):
        self.action_pub = rospy.Publisher(rospy.get_param('control'), Twist, queue_size=1)
    if rospy.has_param('ready'): 
      rospy.Subscriber(rospy.get_param('ready'), Empty, self.ready_callback)
    if rospy.has_param('finished'):
      rospy.Subscriber(rospy.get_param('finished'), Empty, self.finished_callback)
    if rospy.has_param('rgb_image') and not FLAGS.depth_input:
      rospy.Subscriber(rospy.get_param('rgb_image'), Image, self.image_callback)
    if rospy.has_param('depth_image'):
      if FLAGS.depth_input or FLAGS.auxiliary_depth or FLAGS.rl:        
        rospy.Subscriber(rospy.get_param('depth_image'), Image, self.depth_callback)
    if FLAGS.recovery_cameras:
      # callbacks={'left':{'30':image_callback_left_30,'60':image_callback_left_60},'right':{'30':image_callback_right_30,'60':image_callback_right_60}}
      # callbacks_depth={'left':{'30':depth_callback_left_30,'60':depth_callback_left_60},'right':{'30':depth_callback_right_30,'60':depth_callback_right_60}}
      self.recovery_images = {}
      for d in ['left','right']:
        self.recovery_images[d] = {}
        for c in ['30','60']:
          self.recovery_images[d][c]={}
          self.recovery_images[d][c]['rgb']=[]
          self.recovery_images[d][c]['depth']=[]
          rospy.Subscriber(re.sub(r"kinect","kinect_"+d+"_"+c,rospy.get_param('rgb_image')), Image, self.image_callback_recovery, (d, c))
          rospy.Subscriber(re.sub(r"kinect","kinect_"+d+"_"+c,rospy.get_param('depth_image')), Image, self.depth_callback_recovery, (d, c))
      
    if not FLAGS.real: # in simulation
      self.replay_buffer = ReplayBuffer(FLAGS.buffer_size, FLAGS.random_seed)
      self.accumloss = 0
      rospy.Subscriber('/ground_truth/state', Odometry, self.gt_callback)
      
      
  def ready_callback(self,msg):
    if not self.ready and self.finished:
      print('Neural control activated.')
      self.ready = True
      self.start_time = rospy.get_time()
      self.finished = False
      self.exploration_noise.reset()
      self.speed=FLAGS.speed + (not FLAGS.evaluate)*np.random.uniform(-FLAGS.sigma_x, FLAGS.sigma_x)
      if rospy.has_param('evaluate') and not FLAGS.off_policy and not FLAGS.real:
        FLAGS.evaluate = rospy.get_param('evaluate')
        # print '--> set evaluate to: ',FLAGS.evaluate
      # if FLAGS.lstm:
      #   self.state=self.model.get_init_state(True)
      #   print 'set state to: ', self.state  
      if rospy.has_param('world_name') :
        self.world_name = os.path.basename(rospy.get_param('world_name').split('.')[0])
        if 'sandbox' in self.world_name: self.world_name='sandbox'
    
  def gt_callback(self, data):
    if not self.ready: return
    current_pos=[data.pose.pose.position.x,
                    data.pose.pose.position.y,
                    data.pose.pose.position.z,
                    data.pose.pose.orientation.z]
    if len(self.last_position)!= 0:
        self.current_distance += np.sqrt((self.last_position[0]-current_pos[0])**2+(self.last_position[1]-current_pos[1])**2)
    self.furthest_point=max([self.furthest_point, np.sqrt(current_pos[0]**2+current_pos[1]**2)])
      
    self.last_position=current_pos
    # print(self.furthest_point)
  
  def process_rgb(self, msg):
    # self.time_1 = time.time()
    # if not self.ready or self.finished or (rospy.get_time()-self.start_time) < self.delay_evaluation: return
    if not self.ready or self.finished: return []
    try:
      # Convert your ROS Image message to OpenCV2
      im = bridge.imgmsg_to_cv2(msg, 'rgb8') # changed to normal RGB order as i ll use matplotlib and PIL instead of opencv
      # an idea could be to swap these channels during online training as this shouldnt matter though this could
      # explain the performance drop coming from a pretrained network.
      # This does mean that online trained nets might be worth nothing...
      # im = bridge.imgmsg_to_cv2(msg, 'bgr8') 
    except CvBridgeError as e:
      print(e)
    else:
      # self.time_2 = time.time()
      size = self.model.input_size[1:]
      im = sm.imresize(im,tuple(size),'nearest')
      # im = im*1/255.
      # Basic preprocessing: center + make 1 standard deviation
      # im -= FLAGS.mean
      # im = im*1/FLAGS.std
      return im

  def process_depth(self, msg):
    # if not self.ready or self.finished or (rospy.get_time()-self.start_time) < self.delay_evaluation: return
    if not self.ready or self.finished: return [] 
    try:
      # Convert your ROS Image message to OpenCV2
      im = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')#gets float of 32FC1 depth image
    except CvBridgeError as e:
      print(e)
    else:
      im = im[::8,::8]
      shp=im.shape
      # assume that when value is not a number it is due to a too large distance
      # values can be nan for when they are closer than 0.5m but than the evaluate node should
      # kill the run anyway.
      im=np.asarray([ e*1.0 if not np.isnan(e) else 5 for e in im.flatten()]).reshape(shp) # clipping nans: dur: 0.010
      # print 'min: ',np.amin(im),' and max: ',np.amax(im)
      # im=np.asarray([ e*1.0 if not np.isnan(e) else 0 for e in im.flatten()]).reshape(shp) # clipping nans: dur: 0.010
      # Resize image
      if FLAGS.auxiliary_depth or FLAGS.rl:
        size = self.model.depth_input_size #(55,74)
        im=sm.imresize(im,size,'nearest') # dur: 0.002
        # cv2.imshow('depth', im) # dur: 0.002
      if FLAGS.depth_input:
        size = (self.model.input_size[1],self.model.input_size[1])
        im=sm.imresize(im,size,'nearest') # dur: 0.009
        im=im[im.shape[0]/2, :]
        # cv2.imshow('depth', im.reshape(1,im.shape[0])) # dur: 0.002
      # cv2.waitKey(2)
      im = im *1/255.*5. # dur: 0.00004
      return im
    
  def image_callback(self, msg):
    im = self.process_rgb(msg)
    if len(im)!=0: 
      if FLAGS.n_fc:
        self.nfc_images.append(im)
        self.nfc_positions.append(self.last_position[:])
        if len(self.nfc_images) < FLAGS.n_frames:
          # print('filling concatenated frames: ',len(self.nfc_images))
          return
        else:
          # concatenate last n-frames
          im = np.concatenate(np.asarray(self.nfc_images[-FLAGS.n_frames:]),axis=2)
          self.nfc_images = self.nfc_images[-FLAGS.n_frames+1:] # concatenate last n-1-frames

          self.nfc_positions.pop(0) #get rid of the first one
          assert len(self.nfc_positions) == FLAGS.n_frames-1
          self.target_odom = [self.nfc_positions[1][i]-self.nfc_positions[0][i] for i in range(len(self.nfc_positions[0]))]
          # print 'Target odometry: ', self.target_odom 
      self.process_input(im)

  def image_callback_recovery(self, msg, args):
    im = self.process_rgb(msg)
    if len(im)==0: return
    trgt = -100.
    if FLAGS.auxiliary_depth and len(self.recovery_images[args[0]][args[1]]['depth']) == 0:
      print("No target depth: {0} {1}".format(args[0], args[1]))
      return
    else: trgt_depth = copy.deepcopy(self.recovery_images[args[0]][args[1]]['depth'])  
    if len(self.target_control) == 0:
      print("No target control: {0} {1}".format(args[0], args[1]))
      return
    else:
      # left ==> -1, right ==> +1, 30dg ==> 0.5, 60dg ==> 1.0
      compensation = -(args[0]=='left')*int(args[1])/60.+(args[0]=='right')*int(args[1])/60.
      trgt = compensation+self.target_control[5]
    if FLAGS.experience_replay and not FLAGS.evaluate and trgt != -100:
      if FLAGS.auxiliary_depth:
        print('added experience of camera: {0} {1} with control {2}'.format(args[0],args[1],trgt))
        self.replay_buffer.add(im,[trgt],[trgt_depth])
      else:
        self.replay_buffer.add(im,[trgt])
    
  def depth_callback(self, msg):
    im = self.process_depth(msg)
    if len(im)!=0: 
      if FLAGS.auxiliary_depth or FLAGS.rl:
        self.target_depth = im #(64,)
      if FLAGS.depth_input:
        if FLAGS.network == 'nfc_control':
          self.nfc_images.append(im)
          if len(self.nfc_images)<4:
            # print('filling concatenated frames: ',len(self.nfc_images))
            return
          else:
            # print np.asarray(self.nfc_images).shape
            im = np.concatenate(np.asarray(self.nfc_images))
            # print im.shape
            self.nfc_images.pop(0)
        self.process_input(im)
        
  def depth_callback_recovery(self, msg, args):
    im = self.process_depth(msg)
    self.recovery_images[args[0]][args[1]]['depth'] = im
    
  def process_input(self, im):
    self.time_3 = time.time()
    trgt = -100.
    # if self.target_control == None or FLAGS.evaluate:
    if FLAGS.evaluate: ### EVALUATE
      trgt_depth = []
      trgt_odom = []
      with_loss = False
      if len(self.target_control)!=0 and not FLAGS.auxiliary_depth and not FLAGS.auxiliary_odom: 
        trgt = self.target_control[5]
        with_loss = True
      elif len(self.target_control)!=0 and FLAGS.auxiliary_depth and len(self.target_depth)!=0 and not FLAGS.auxiliary_odom: 
        trgt = self.target_control[5]
        trgt_depth = [copy.deepcopy(self.target_depth)]
        with_loss = True
      elif len(self.target_control)!=0 and not FLAGS.auxiliary_depth and FLAGS.auxiliary_odom and len(self.target_odom)!=0: 
        trgt = self.target_control[5]
        trgt_odom = [copy.deepcopy(self.target_odom)]
        with_loss = True
      elif len(self.target_control)!=0 and FLAGS.auxiliary_depth and len(self.target_depth)!=0 and FLAGS.auxiliary_odom and len(self.target_odom)!=0: 
        trgt = self.target_control[5]
        trgt_odom = [copy.deepcopy(self.target_odom)]
        trgt_depth = [copy.deepcopy(self.target_depth)]
        with_loss = True
      if with_loss and False: # for now skip calculating accumulated loses.
        prev_ctr = [[self.prev_control[0]]]
        control, self.state, losses, aux_results = self.model.forward([[im]] if FLAGS.lstm else [im], states=self.state, 
          auxdepth=FLAGS.show_depth, auxodom=FLAGS.show_odom, prev_action=prev_ctr, 
          targets=[[trgt]], target_depth=trgt_depth, target_odom=trgt_odom)
        if len(self.accumlosses.keys())==0: 
          self.accumlosses = losses
        else: 
          # self.accumlosses=[self.accumlosses[i]+losses[i] for i in range(len(losses))]
          for v in losses.keys(): self.accumlosses[v]=self.accumlosses[v]+losses[v]
      else:
        prev_ctr = [[self.prev_control[0]]]
        control, self.state, losses, aux_results = self.model.forward([[im]] if FLAGS.lstm else [im], states=self.state, auxdepth=FLAGS.show_depth, 
          auxodom=FLAGS.show_odom, prev_action=prev_ctr)
      if FLAGS.show_depth and FLAGS.auxiliary_depth and len(aux_results)>0: self.aux_depth = aux_results.pop(0)
      if FLAGS.show_odom and FLAGS.auxiliary_odom and len(aux_results)>0: self.aux_odom = aux_results.pop(0)
    
    else: ###TRAINING
      # Get necessary labels, if label is missing wait...
      if len(self.target_control) == 0:
        print('No target control')
        return
      else:
        trgt = self.target_control[5]
        # print(trgt)
      if (FLAGS.auxiliary_depth or FLAGS.rl) and len(self.target_depth) == 0:
        print('No target depth')
        return
      else:
        trgt_depth = copy.deepcopy(self.target_depth)
        # self.target_depth = []
      if FLAGS.auxiliary_odom and (len(self.target_odom) == 0 or len(self.prev_control) == 0):
        print('no target odometry or previous control')
        return
      else:
        trgt_odom = copy.deepcopy(self.target_odom)
      # check if depth image corresponds to rgb image
      # cv2.imshow('rgb', im)
      # cv2.waitKey(2)
      # cv2.imshow('depth', trgt_depth*1/5.)
      # cv2.waitKey(2)
      # ---------------------------------------------------------- DEPRECATED
      # if not FLAGS.experience_replay: ### TRAINING WITHOUT EXPERIENCE REPLAY 
      #   if FLAGS.auxiliary_depth:
      #     control, losses = self.model.backward([im],[[trgt]], [[[trgt_depth]]])
      #   else:
      #     control, losses = self.model.backward([im],[[trgt]])
      #   print 'Difference: '+str(control[0,0])+' and '+str(trgt)+'='+str(abs(control[0,0]-trgt))
      #   self.accumlosses += losses[0]
      # else: ### TRAINING WITH EXPERIENCE REPLAY
      # wait for first target depth in case of auxiliary depth.
      # in case the network can predict the depth
      self.time_4 = time.time()
      prev_ctr = [[self.prev_control[0]]]

      control, self.state, losses, aux_results = self.model.forward([[im]] if FLAGS.lstm else [im], states=self.state , 
        auxdepth=FLAGS.show_depth, auxodom=FLAGS.show_odom, prev_action=prev_ctr)
      if FLAGS.show_depth: self.aux_depth = aux_results.pop(0)
      if FLAGS.show_odom: self.aux_odom = aux_results.pop(0)
      self.time_5 = time.time()
      # print 'state: ', self.state
    ### SEND CONTROL
    noise = self.exploration_noise.noise()
    # yaw = control[0,0]
    # if np.random.binomial(1,FLAGS.epsilon) and not FLAGS.evaluate:
    # yaw = max(-1,min(1,np.random.normal()))
    if trgt != 100 and not FLAGS.evaluate:
      action = trgt if np.random.binomial(1,FLAGS.alpha**self.run) else control[0,0]
    else:
      action = control[0,0]
    msg = Twist()
    if FLAGS.type_of_noise == 'ou':
      msg.linear.x = self.speed #0.8 # 1.8 #
      # msg.linear.x = FLAGS.speed+(not FLAGS.evaluate)*FLAGS.sigma_x*noise[0] #0.8 # 1.8 #
      msg.linear.y = (not FLAGS.evaluate)*noise[1]*FLAGS.sigma_y
      msg.linear.z = (not FLAGS.evaluate)*noise[2]*FLAGS.sigma_z
      msg.angular.z = max(-1,min(1,action+(not FLAGS.evaluate)*FLAGS.sigma_yaw*noise[3]))
    elif FLAGS.type_of_noise == 'uni':
      msg.linear.x = self.speed
      # msg.linear.x = FLAGS.speed + (not FLAGS.evaluate)*np.random.uniform(-FLAGS.sigma_x, FLAGS.sigma_x)
      msg.linear.y = (not FLAGS.evaluate)*np.random.uniform(-FLAGS.sigma_y, FLAGS.sigma_y)
      msg.linear.z = (not FLAGS.evaluate)*np.random.uniform(-FLAGS.sigma_z, FLAGS.sigma_z)
      msg.angular.z = max(-1,min(1,action+(not FLAGS.evaluate)*np.random.uniform(-FLAGS.sigma_yaw, FLAGS.sigma_yaw)))
    else:
      raise IOError( 'Type of noise is unknown: {}'.format(FLAGS.type_of_noise))
    self.action_pub.publish(msg)
    self.prev_control = [msg.angular.z]
    self.time_6 = time.time()
    if FLAGS.show_depth and len(self.aux_depth) != 0 and not self.finished:
      # print('shape aux depth: {}'.format(self.aux_depth.shape))
      self.aux_depth = self.aux_depth.flatten()
      self.depth_pub.publish(self.aux_depth)
      self.aux_depth = []
    if FLAGS.show_odom and len(self.aux_odom) != 0 and not self.finished and len(trgt_odom)!=0:
      # debug odom by checking image + odometry correspondences:
      # final_img = cv2.hconcat((im[:,:,0:3], im[:,:,3:6],im[:,:,6:]))
      # final_img = cv2.hconcat((im[:,:,[2,1,0]], im[:,:,[5,4,3]],im[:,:,[8,7,6]]))
      # print trgt_odom
      # cv2.imshow('Final', final_img)      
      # cv2.waitKey(100)
      # cv2.destroyAllWindows()        
      concat_odoms=np.concatenate((self.aux_odom.astype(np.float32).flatten(), np.array(trgt_odom).astype(np.float32).flatten()))
      # self.odom_pub.publish(self.aux_odom.flatten())
      # print concat_odoms[4:6],' and ',concat_odoms[0:2]
      self.odom_pub.publish(concat_odoms.astype(np.float32))
      self.odom_error.append(np.abs(np.array(trgt_odom).flatten()-self.aux_odom.flatten()))
      self.aux_odom = []
      
    # ADD EXPERIENCE REPLAY
    if FLAGS.experience_replay and not FLAGS.evaluate and trgt != -100:
      aux_info = {}
      if FLAGS.auxiliary_depth or FLAGS.rl: 
        aux_info['target_depth']=trgt_depth
      if FLAGS.auxiliary_odom: 
        aux_info['target_odom']=trgt_odom
        aux_info['prev_action']=prev_ctr
      if FLAGS.lstm:
        # aux_info['state']=(np.zeros(()))
        # state type:  <type 'tuple'>  len:  2  len sub:  2  len subsub:  1  len subsubsub:  100
        aux_info['state']=self.state
        # aux_info['state']=((np.zeros((1,100)),np.zeros((1,100))+10),(np.ones((1,100)),np.ones((1,100))+20))
        # print aux_info['state']
        # (state layer0,output layer0,state layer1,output layer1)
        # print 'state type: ',type(aux_info['state']),' len: ', len(aux_info['state']),' len sub: ', len(aux_info['state'][0]),' len subsub: ', len(aux_info['state'][0][0]),' len subsubsub: ', len(self.state[0][0][0])
      self.replay_buffer.add(im,[trgt],aux_info=aux_info)
      
    self.time_7 = time.time()
    if FLAGS.save_input: 
      self.depthfile = open(self.logfolder+'/depth_input', 'a')
      np.set_printoptions(precision=5)
      message="{0} : {1} : {2:.4f} \n".format(self.run, ' '.join('{0:.5f}'.format(k) for k in np.asarray(im)), trgt)
      self.depthfile.write(message)
      self.depthfile.close()
    self.time_8 = time.time()
    # print 'processed image @: {0:.2f}'.format(time.time())
    
    # print("Time debugging: \n cvbridge: {0} , \n resize: {1}, \n copy: {2} , \n net pred: {3}, \n pub: {4},\n exp buf: {5},\n pos file: {6} s".format((self.time_2-self.time_1),
      # (self.time_3-self.time_2),(self.time_4-self.time_3),(self.time_5-self.time_4),(self.time_6-self.time_5),(self.time_7-self.time_6),(self.time_8-self.time_7)))
    # Delay values with auxiliary depth (at the beginning of training)
    # cv bridge (RGB): 0.0003s
    # resize (RGB): 0.0015s
    # copy control+depth: 2.7e-5 s
    # net prediction: 0.011s
    # publication: 0.0002s
    # fill experience buffer: 1.8e-5 s
    # write position: 2.1e-6 s

  def supervised_callback(self, data):
    if not self.ready: return
    self.target_control = [data.linear.x,
      data.linear.y,
      data.linear.z,
      data.angular.x,
      data.angular.y,
      data.angular.z]
      
  def finished_callback(self,msg):
    if self.ready and not self.finished:
      # self.depth_pub.publish(self.aux_depth)
      print('neural control deactivated.')
      self.ready=False
      self.finished=True
      # Train model from experience replay:
      # Train the model with batchnormalization out of the image callback loop
      activation_images = []
      depth_predictions = []
      endpoint_activations = []
      tloss = [] #total loss
      closs = [] #control loss
      dloss = [] #depth loss
      oloss = [] #odometry loss
      qloss = [] #RL cost-to-go loss
      tlossm, clossm, dlossm, olossm, qlossm, tlossm_eva, clossm_eva, dlossm_eva, olossm_eva, qlossm_eva = 0,0,0,0,0,0,0,0,0,0
      #tot_batch_loss = []
      if FLAGS.experience_replay and self.replay_buffer.size()>(FLAGS.batch_size if not FLAGS.lstm else FLAGS.batch_size*FLAGS.num_steps) and not FLAGS.evaluate:
        for b in range(min(int(self.replay_buffer.size()/FLAGS.batch_size), 10)):
          inputs, targets, aux_info = self.replay_buffer.sample_batch(FLAGS.batch_size)
          # import pdb; pdb.set_trace()
          #print('time to smaple batch of images: ',time.time()-st)
          if b==0:
            if FLAGS.plot_activations:
              activation_images= self.model.plot_activations(inputs, targets.reshape((-1,1)))
            if FLAGS.plot_depth and FLAGS.auxiliary_depth:
              depth_predictions = self.model.plot_depth(inputs, aux_info['target_depth'].reshape(-1,55,74))
            if FLAGS.plot_histograms:
              endpoint_activations = self.model.get_endpoint_activations(inputs)
          init_state=[]
          depth_targets=[]
          odom_targets=[]
          prev_action=[]
          if FLAGS.lstm:
            init_state=(aux_info['state'][:,0,0,0,0,:],
                        aux_info['state'][:,0,0,1,0,:],
                        aux_info['state'][:,0,1,0,0,:],
                        aux_info['state'][:,0,1,1,0,:])
            # if FLAGS.use_init_state:
            #   init_state=
            assert init_state[0].shape[0]==FLAGS.batch_size
            # print 'init_state sizes ',init_state[0].shape
          if FLAGS.auxiliary_depth or FLAGS.rl: 
            depth_targets=aux_info['target_depth'].reshape(-1,55,74)
            # depth_targets=aux_info['target_depth'].reshape(-1,55,74) if not FLAGS.lstm else aux_info['target_depth'].reshape(-1,FLAGS.num_steps, 55,74)
          if FLAGS.auxiliary_odom: 
            odom_targets=aux_info['target_odom'].reshape(-1,4) if not FLAGS.lstm else aux_info['target_odom'].reshape(-1,FLAGS.num_steps, 4)
            prev_action=aux_info['prev_action'].reshape(-1,1) #if not FLAGS.lstm else aux_info['prev_action'].reshape(-1,FLAGS.num_steps, 1)
          # todo add initial state for each rollout in the batch
          controls, losses = self.model.backward(inputs,init_state,targets[:].reshape(-1,1),depth_targets, odom_targets, prev_action)
          tloss = losses['t']
          if not FLAGS.rl or FLAGS.auxiliary_ctr: closs = losses['c']
          if FLAGS.auxiliary_depth: dloss.append(losses['d'])
          if FLAGS.auxiliary_odom: oloss.append(losses['o'])
          if FLAGS.rl : qloss.append(losses['q'])
        tlossm = np.mean(tloss)
        clossm = np.mean(closs) if not FLAGS.rl or FLAGS.auxiliary_ctr else 0
        dlossm = np.mean(dloss) if FLAGS.auxiliary_depth else 0
        olossm = np.mean(oloss) if FLAGS.auxiliary_odom else 0
        qlossm = np.mean(qloss) if FLAGS.rl else 0
      else:
        print('Evaluating or filling buffer or no experience_replay: ', self.replay_buffer.size())
        if 't' in self.accumlosses.keys() : tlossm_eva = self.accumlosses['t']
        if 'c' in self.accumlosses.keys() : clossm_eva = self.accumlosses['c']
        if 'd' in self.accumlosses.keys() : dlossm_eva = self.accumlosses['d']
        if 'o' in self.accumlosses.keys() : olossm_eva = self.accumlosses['o']
        if 'q' in self.accumlosses.keys() : qlossm_eva = self.accumlosses['q']

      if not FLAGS.evaluate:
        self.average_distance = self.average_distance-self.average_distance/(self.run+1)
        self.average_distance = self.average_distance+self.current_distance/(self.run+1)
      else:
        self.average_distance_eva = self.average_distance_eva-self.average_distance_eva/(self.run_eva+1)
        self.average_distance_eva = self.average_distance_eva+self.current_distance/(self.run_eva+1)
      
      odom_errx, odom_erry, odom_errz, odom_erryaw = 0,0,0,0
      if len(self.odom_error) != 0:
        odom_errx=np.mean([e[0] for e in self.odom_error])
        odom_erry=np.mean([e[1] for e in self.odom_error])
        odom_errz=np.mean([e[2] for e in self.odom_error])
        odom_erryaw=np.mean([e[3] for e in self.odom_error])
      try:
        sumvar={}
        # sumvar={k : 0 for k in self.model.summary_vars.keys()}
        sumvar["Distance_current_"+self.world_name if len(self.world_name)!=0 else "Distance_current"]=self.current_distance
        sumvar["Distance_furthest_"+self.world_name if len(self.world_name)!=0 else "Distance_furthest"]=self.furthest_point
        if FLAGS.evaluate:
          sumvar["Distance_average_eva"]=self.average_distance_eva
        else:
          sumvar["Distance_average"]=self.average_distance
        if tlossm != 0 : sumvar["Loss_total"]=tlossm
        if clossm != 0 : sumvar["Loss_control"]=clossm 
        if dlossm != 0 : sumvar["Loss_depth"]=dlossm 
        if olossm != 0 : sumvar["Loss_odom"]=olossm 
        if qlossm != 0 : sumvar["Loss_q"]=qlossm 
        if tlossm_eva != 0 : sumvar["Loss_total_eva"]=tlossm_eva
        if clossm_eva != 0 : sumvar["Loss_control_eva"]=clossm_eva 
        if dlossm_eva != 0 : sumvar["Loss_depth_eva"]=dlossm_eva 
        if olossm_eva != 0 : sumvar["Loss_odom_eva"]=olossm_eva 
        if qlossm_eva != 0 : sumvar["Loss_q_eva"]=qlossm_eva 
        if odom_errx != 0 : sumvar["odom_errx"]=odom_errx 
        if odom_erry != 0 : sumvar["odom_erry"]=odom_erry 
        if odom_errz != 0 : sumvar["odom_errz"]=odom_errz 
        if odom_erryaw != 0 : sumvar["odom_erryaw"]=odom_erryaw
        if FLAGS.plot_activations and len(activation_images)!=0:
          sumvar["conv_activations"]=activation_images
          # sumvar.append(activation_images)
        if FLAGS.plot_depth and FLAGS.auxiliary_depth:
          sumvar["depth_predictions"]=depth_predictions
          # sumvar.append(depth_predictions)
        if FLAGS.plot_histograms:
          for i, ep in enumerate(self.model.endpoints):
            sumvar['activations_{}'.format(ep)]=endpoint_activations[i]
          # sumvar.extend(endpoint_activations)
        self.model.summarize(sumvar)
      except Exception as e:
        print('failed to write', e)
        pass
      else:
        print('{0}: control finished {1}:[ current_distance: {2:0.3f}, average_distance: {3:0.3f}, furthest point: {4:0.1f}, total loss: {5:0.3f}, control loss: {6:0.3f}, depth loss: {7:0.3f}, odom loss: {8:0.3f}, q loss: {9:0.3f}, world: {10}'.format(time.strftime('%H:%M'), 
          self.run if not FLAGS.evaluate else self.run_eva, self.current_distance, self.average_distance if not FLAGS.evaluate else self.average_distance_eva, 
          self.furthest_point, tlossm if not FLAGS.evaluate else tlossm_eva, clossm if not FLAGS.evaluate else clossm_eva, dlossm if not FLAGS.evaluate else dlossm_eva, olossm if not FLAGS.evaluate else olossm_eva, 
          qlossm if not FLAGS.evaluate else qlossm_eva, self.world_name))
        l_file = open(self.logfile,'a')
        tag='train'
        if FLAGS.evaluate:
          tag='val'
        l_file.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}\n'.format(
            self.run if not FLAGS.evaluate else self.run_eva, 
          self.current_distance, 
          self.average_distance if not FLAGS.evaluate else self.average_distance_eva, 
          self.furthest_point, 
          tlossm,
          clossm,
          dlossm,
          olossm,
          qlossm,
          tag,
          self.world_name))
        l_file.close()
      self.accumlosses = {}
      self.maxy = -10
      self.current_distance = 0
      self.last_position = []
      self.nfc_images = []
      self.nfc_positions = []
      self.furthest_point = 0
      if FLAGS.lstm and not FLAGS.evaluate: self.replay_buffer.new_run()
      self.world_name = ''
      if self.run%10==0 and not FLAGS.evaluate:
        # Save a checkpoint every 20 runs.
        self.model.save(self.logfolder)
      self.state=[]
      if not FLAGS.evaluate:
        self.run+=1  
      else :
        self.run_eva+=1
      # wait for gzserver to be killed
      gzservercount=1
      while gzservercount > 0:
        #print('gzserver: ',gzservercount)
        gzservercount = os.popen("ps -Af").read().count('gzserver')
        time.sleep(0.1)
      sys.stdout.flush()

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
  
