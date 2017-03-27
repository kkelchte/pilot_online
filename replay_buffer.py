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
        self.buffer = []
        
        self.buffer = deque()
        random.seed(random_seed)
        # np.random.seed(random_seed)

    def add(self, state, target, aux_target=None):
        experience = (state, target)
        if aux_target:
            experience = (state, target, aux_target)
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            # self.buffer.pop(0)
            self.buffer.append(experience)

    def size(self):
        return self.count
      
    def softmax(self, x):
      e_x = np.exp(x-np.max(x))
      return e_x/e_x.sum()

    def sample_batch(self, batch_size):
        #batch = []
        if self.count < batch_size:
          # Add different distribution: exponentially/gaussian decaying over time.
          batch = random.sample(self.buffer, self.count)
          # if FLAGS.weight_replay:
          #   inds = np.random.choice(range(self.count), self.count, p=list(self.softmax(2.5*np.log(range(1,1+self.count)))))
          # else:
          #   inds = np.random.choice(range(self.count), self.count)
        else:
          batch = random.sample(self.buffer, batch_size)
          # if FLAGS.weight_replay:
          #   inds = np.random.choice(range(batch_size), batch_size, p=list(self.softmax(2.5*np.log(range(1,1+batch_size)))))
          # else:
          #   inds = np.random.choice(range(batch_size), batch_size)
        # batch = [ self.buffer[i] for i in inds]
        
        state_batch = np.array([_[0] for _ in batch])
        target_batch = np.array([_[1] for _ in batch])
        aux_batch = None
        if len(batch[0]) == 3:
            aux_batch = np.array([_[2] for _ in batch])

        return state_batch, target_batch, aux_batch

    def clear(self):
        self.deque.clear()
        self.count = 0
