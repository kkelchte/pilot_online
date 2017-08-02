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
tf.app.flags.DEFINE_integer("batch_size", 16, "Define the size of minibatches.")
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

tf.app.flags.DEFINE_boolean("off_policy",False,"In case the network is off_policy, the control is published on supervised_vel instead of cmd_vel.")
tf.app.flags.DEFINE_boolean("show_depth",False,"Publish the predicted horizontal depth array to topic ./depth_prection so show_depth can visualize this in another node.")
tf.app.flags.DEFINE_boolean("recovery_cameras",False,"Listen to recovery cameras (left-right 30-60) and add them in replay buffer.")
tf.app.flags.DEFINE_boolean("save_input",False,"Write depth input to file in order to check values later.")

tf.app.flags.DEFINE_float("ou_theta", 0.15, "Epsilon is the probability that the control is picked randomly.")
# tf.app.flags.DEFINE_float("ou_sigma", 0.3, "Alpha is the amount of noise in the general y, z and Y direction during training to ensure it visits the whole corridor.")
# =================================================

launch_popen=None

class PilotNode(object):
  
  def __init__(self, model, logfolder):
    print('initialize pilot node')

    # Initialize replay memory
    self.logfolder = logfolder
    self.world_name = ''
    self.logfile = logfolder+'/tensorflow_log'
    self.run=0
    self.maxy=-10
    self.current_distance=0
    self.average_distance=0
    self.furthest_point=0
    self.last_position=[]
    self.model = model
    self.ready=False 
    self.finished=True
    self.target_control = []
    self.target_depth = []
    self.aux_depth = []
    self.nfc_images =[] #used by n_fc networks for building up concatenated frames
    rospy.init_node('pilot', anonymous=True)
    self.exploration_noise = OUNoise(4, 0, FLAGS.ou_theta,1)

    # self.delay_evaluation = 5 #can't be set by ros because node is started before ros is started...
    if FLAGS.show_depth:
      self.depth_pub = rospy.Publisher('/depth_prediction', numpy_msg(Floats), queue_size=1)
    if FLAGS.off_policy:
      self.action_pub = rospy.Publisher('/supervised_vel', Twist, queue_size=1)
      if rospy.has_param('control'):
        rospy.Subscriber(rospy.get_param('control'), Twist, self.supervised_callback)
    else:
      rospy.Subscriber('/supervised_vel', Twist, self.supervised_callback)
      if rospy.has_param('control'):
        self.action_pub = rospy.Publisher(rospy.get_param('control'), Twist, queue_size=1)
    ### TO BE CLEANED UP
    if FLAGS.real:
      self.action_pub=rospy.Publisher('/pilot_vel', Twist, queue_size=1)

    if rospy.has_param('ready'): 
      rospy.Subscriber(rospy.get_param('ready'), Empty, self.ready_callback)
    if rospy.has_param('finished'):
      rospy.Subscriber(rospy.get_param('finished'), Empty, self.finished_callback)
    if rospy.has_param('rgb_image') and not FLAGS.depth_input:
      rospy.Subscriber(rospy.get_param('rgb_image'), Image, self.image_callback)
    if rospy.has_param('depth_image'):
      if FLAGS.depth_input:        
        rospy.Subscriber(rospy.get_param('depth_image'), Image, self.depth_callback)
      if FLAGS.auxiliary_depth:
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
      if rospy.has_param('world_name') :
        self.world_name = os.path.basename(rospy.get_param('world_name').split('.')[0])
    
  def gt_callback(self, data):
    if not self.ready: return
    current_pos=[data.pose.pose.position.x,
                    data.pose.pose.position.y,
                    data.pose.pose.position.z]
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
      im = bridge.imgmsg_to_cv2(msg, 'bgr8') 
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
      im=np.asarray([ e*1.0 if not np.isnan(e) else 0 for e in im.flatten()]).reshape(shp) # clipping nans: dur: 0.010
      # Resize image
      if FLAGS.auxiliary_depth:
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
        if len(self.nfc_images) < FLAGS.n_frames:
          print('filling concatenated frames: ',len(self.nfc_images))
          return
        else:
          im = np.concatenate(np.asarray(self.nfc_images),axis=2)
          print im.shape
          self.nfc_images.pop(0)  
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
      if FLAGS.auxiliary_depth:
        self.target_depth = im #(64,)
        
  def depth_callback_recovery(self, msg, args):
    im = self.process_depth(msg)
    self.recovery_images[args[0]][args[1]]['depth'] = im
    
  def process_input(self, im):
    self.time_3 = time.time()
    trgt = -100.
    # if self.target_control == None or FLAGS.evaluate:
    if FLAGS.evaluate: ### EVALUATE
      control, self.aux_depth = self.model.forward([im], aux=FLAGS.show_depth)
    else: ###TRAINING
      # Get necessary labels, if label is missing wait...
      if len(self.target_control) == 0:
        print('No target control')
        return
      else:
        trgt = self.target_control[5]
        # print(trgt)
      if FLAGS.auxiliary_depth and len(self.target_depth) == 0:
        print('No target depth')
        return
      else:
        trgt_depth = copy.deepcopy(self.target_depth)
        # self.target_depth = []
      # check if depth image corresponds to rgb image
      # cv2.imshow('rgb', im)
      # cv2.waitKey(2)
      # cv2.imshow('depth', trgt_depth*1/5.)
      # cv2.waitKey(2)
      if not FLAGS.experience_replay: ### TRAINING WITHOUT EXPERIENCE REPLAY 
        if FLAGS.auxiliary_depth:
          control, losses = self.model.backward([im],[[trgt]], [[[trgt_depth]]])
        else:
          control, losses = self.model.backward([im],[[trgt]])
        print 'Difference: '+str(control[0,0])+' and '+str(trgt)+'='+str(abs(control[0,0]-trgt))
        self.accumloss += losses[0]
      else: ### TRAINING WITH EXPERIENCE REPLAY
        # wait for first target depth in case of auxiliary depth.
        # in case the network can predict the depth
        self.time_4 = time.time()
        control, self.aux_depth = self.model.forward([im], aux=FLAGS.show_depth)
        # print('control: {}'.format(control))
        self.time_5 = time.time()
    # import pdb; pdb.set_trace()
    ### SEND CONTROL
    noise = self.exploration_noise.noise()
    # yaw = control[0,0]
    # if np.random.binomial(1,FLAGS.epsilon) and not FLAGS.evaluate:
    # yaw = max(-1,min(1,np.random.normal()))
    msg = Twist()
    if FLAGS.type_of_noise == 'ou':
      msg.linear.x = FLAGS.speed+(not FLAGS.evaluate)*FLAGS.sigma_x*noise[0] #0.8 # 1.8 #
      msg.linear.y = (not FLAGS.evaluate)*noise[1]*FLAGS.sigma_y
      msg.linear.z = (not FLAGS.evaluate)*noise[2]*FLAGS.sigma_z
      msg.angular.z = max(-1,min(1,control[0,0]+FLAGS.sigma_yaw*noise[3]))
    elif FLAGS.type_of_noise == 'uni':
      msg.linear.x = FLAGS.speed + (not FLAGS.evaluate)*np.random.uniform(-FLAGS.sigma_x, FLAGS.sigma_x)
      msg.linear.y = (not FLAGS.evaluate)*np.random.uniform(-FLAGS.sigma_y, FLAGS.sigma_y)
      msg.linear.z = (not FLAGS.evaluate)*np.random.uniform(-FLAGS.sigma_z, FLAGS.sigma_z)
      msg.angular.z = max(-1,min(1,control[0,0]+(not FLAGS.evaluate)*np.random.uniform(-FLAGS.sigma_yaw, FLAGS.sigma_yaw)))
    else:
      raise IOError( 'Type of noise is unknown: {}'.format(FLAGS.type_of_noise))
    self.action_pub.publish(msg)
    self.time_6 = time.time()
    if FLAGS.show_depth and len(self.aux_depth) != 0 and not self.finished:
      # print('shape aux depth: {}'.format(self.aux_depth.shape))
      self.aux_depth = self.aux_depth.flatten()
      # self.ready=False
      # import pdb; pdb.set_trace
      self.depth_pub.publish(self.aux_depth)
      self.aux_depth = []
      # import pdb; pdb.set_trace()
    # ADD EXPERIENCE REPLAY
    if FLAGS.experience_replay and not FLAGS.evaluate and trgt != -100:
      if FLAGS.auxiliary_depth:
        self.replay_buffer.add(im,[trgt],[trgt_depth])
      else:
        self.replay_buffer.add(im,[trgt])
    self.time_7 = time.time()
    if FLAGS.save_input: 
      self.depthfile = open(self.logfolder+'/depth_input', 'a')
      np.set_printoptions(precision=5)
      message="{0} : {1} : {2:.4f} \n".format(self.run, ' '.join('{0:.5f}'.format(k) for k in np.asarray(im)), trgt)
      self.depthfile.write(message)
      self.depthfile.close()
    self.time_8 = time.time()
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
      #tot_batch_loss = []
      if FLAGS.experience_replay and self.replay_buffer.size()>FLAGS.batch_size:
        for b in range(min(int(self.replay_buffer.size()/FLAGS.batch_size), 10)):
          #im_b, target_b = self.replay_buffer.sample_batch(FLAGS.batch_size)
          batch = self.replay_buffer.sample_batch(FLAGS.batch_size)
          #print('time to smaple batch of images: ',time.time()-st)
          if b==0:
            if FLAGS.plot_activations:
              activation_images= self.model.plot_activations(batch[0], batch[1].reshape((-1,1)))
            if FLAGS.plot_depth and FLAGS.auxiliary_depth:
              depth_predictions = self.model.plot_depth(batch[0], batch[2][:].reshape(-1,55,74))
            if FLAGS.plot_histograms:
              endpoint_activations = self.model.get_endpoint_activations(batch[0])
          # if FLAGS.evaluate:
          #   # shape control (16,1)
          #   controls, loss = self.model.forward(batch[0],batch[1][:,0].reshape(-1,1))
          #   losses = [loss, 0, 0]
          # else:
          if FLAGS.auxiliary_depth:
            # controls, losses = self.model.backward(batch[0],batch[1][:].reshape(-1,1),batch[2][:].reshape(-1,1,1,64))
            # import pdb; pdb.set_trace()
            controls, losses = self.model.backward(batch[0],batch[1][:].reshape(-1,1),batch[2][:].reshape(-1,55,74))
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
      
      self.average_distance = self.average_distance-self.average_distance/(self.run+1)
      self.average_distance = self.average_distance+self.current_distance/(self.run+1)
      
      try:
        sumvar = [self.accumloss, self.current_distance, tloss, closs, dloss, self.furthest_point]
        if FLAGS.plot_activations and len(activation_images)!=0:
          sumvar.append(activation_images)
        if FLAGS.plot_depth and FLAGS.auxiliary_depth:
          sumvar.append(depth_predictions)
        if FLAGS.plot_histograms:
          sumvar.extend(endpoint_activations)
        self.model.summarize(sumvar)
      except Exception as e:
        print('failed to write', e)
        pass
      else:
        print('{0}: control finished {1}:[ acc loss: {2:0.3f}, current_distance: {3:0.3f}, average_distance: {4:0.3f}, total loss: {5:0.3f}, control loss: {6:0.3f}, depth loss: {7:0.3f}, furthest point: {8:0.1f}, world: {9}'.format(time.strftime('%H:%M'), self.run, self.accumloss, self.current_distance, self.average_distance, tloss, closs, dloss, self.furthest_point, self.world_name))
        l_file = open(self.logfile,'a')
        tag='train'
        if FLAGS.evaluate:
          tag='val'
        l_file.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}\n'.format(
            self.run, 
          self.accumloss, 
          self.current_distance, 
          self.average_distance, 
          tloss, closs, dloss, 
          self.furthest_point, tag, 
          self.world_name))
        l_file.close()
      self.accumloss = 0
      self.maxy = -10
      self.current_distance = 0
      self.last_position = []
      self.furthest_point = 0
      self.world_name = ''
      if self.run%2==0 and not FLAGS.evaluate:
        # Save a checkpoint every 20 runs.
        self.model.save(self.logfolder)
      
      self.run+=1 
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
  
