""" 
Data structure for implementing experience replay
Author: Patrick Emami
"""
from collections import deque
import random
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean("weight_replay", False, "Let more recent experience be more likely to be sampled from the batch.")


class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences 
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer_list = []
        self.current_buffer = deque()
        self.buffer_list.append(self.current_buffer)
        random.seed(random_seed)
        self.num_steps = FLAGS.num_steps if FLAGS.lstm else 1
        # np.random.seed(random_seed)

    def add(self, state, target, aux_info={}):
        experience = (state, target, aux_info)
        if self.count < self.buffer_size: 
            self.current_buffer.append(experience)
            self.count += 1
        else:
          # Get rid of oldest buffer/run once it is smaller than number of steps
          if len(self.buffer_list[0])<=self.num_steps:
            self.count-=len(self.buffer_list.pop(0))
            self.count+=1
          else:
            self.buffer_list[0].popleft()
          # self.buffer.pop(0)
          self.current_buffer.append(experience)
        # print 'buffer size: ',self.count
        # for i,b in enumerate(self.buffer_list): print ' buff: ',i,' len ',len(b)
    def size(self):
        return self.count
    
    def new_run(self):
      self.current_buffer = deque()
      self.buffer_list.append(self.current_buffer)

    def softmax(self, x):
      e_x = np.exp(x-np.max(x))
      return e_x/e_x.sum()

    def sample_batch(self, batch_size):
      assert batch_size < self.count, IOError('batchsize ',batch_size,' is bigger than buffer size: ',self.count)
      # if FLAGS.weight_replay:
      #   inds = np.random.choice(range(batch_size), batch_size, p=list(self.softmax(2.5*np.log(range(1,1+batch_size)))))
      # else:
      #   inds = np.random.choice(range(batch_size), batch_size)
      if FLAGS.lstm:
        # 1. select rollouts / buffers in bufferlist to sample from
        if len(self.buffer_list)>batch_size:
          # caution this could demand big amount of RAM if replay buffer gets big
          selected_buffers = random.sample(self.buffer_list, batch_size)
        else:
          selected_buffers = [random.choice(self.buffer_list) for _ in range(batch_size)]
        # print(selected_buffers)
        # 2. Choose a startindex that allows num_steps concatenated frames and put them in a batch
        batch=[]
        for buff in selected_buffers:
          if len(buff) < self.num_steps: continue
          start_i = random.choice(range(len(buff)-self.num_steps+1))
          batch.append([buff[start_i+i] for i in range(self.num_steps)])
        assert len(batch) != 0, IOError('No buffer in bufferlist(',str(self.buffer_list),') found that is longer than num_steps(',self.num_steps,')')
        # print 'batch ',batch
        # 3. Split up the batch according to input, target and auxiliary information
        # creating an array over time for each rollout and put them together in a list
        input_batch = []
        target_batch = []
        aux_batch = {}
        for k in batch[0][0][2].keys(): aux_batch[k]=[]
        for rollout in batch:
          input_batch.append(np.array([_[0] for _ in rollout]))
          target_batch.append(np.array([_[1] for _ in rollout]))
          for k in rollout[0][2].keys():
            aux_batch[k].append(np.array([_[2][k] for _ in rollout]))
        # 4. Put the list together in an array
        input_batch = np.asarray(input_batch)
        target_batch = np.asarray(target_batch)
        for k in aux_batch.keys(): aux_batch[k]=np.array(aux_batch[k])
      else:
        batch=random.sample(self.current_buffer, batch_size)      
        input_batch = np.array([_[0] for _ in batch])
        target_batch = np.array([_[1] for _ in batch])
        aux_batch = {}
        for k in batch[0][2].keys():
          aux_batch[k]=np.array([_[2][k] for _ in batch])

      return input_batch, target_batch, aux_batch

    def clear(self):
        self.current_buffer.clear()
        self.buffer_list = []
        self.count = 0
