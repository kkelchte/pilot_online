
"""
This code is having aid functions for preparing the data in pilot_data.
Functions:
    - calculate_mean_variance calculates the mean and the variance of the data, used by pilot_data for normalizing the data.
    
"""
import os
import re
import logging
import numpy as np
from os import listdir
from os.path import isfile,join
import time
from PIL import Image
import random

data_path='/home/klaas/pilot_data/sandbox/'
images = [join(data_path, d, 'RGB', img) for d in listdir(data_path) if os.path.isdir(join(data_path,d)) for img in listdir(join(data_path, d, 'RGB')) if isfile(join(data_path, d, 'RGB', img))]
random.shuffle(images)
cnt=0
means = []
stds = []
for imgp in images:
  cnt+=1
  print(cnt, 'out of ', len(images))
  im = Image.open(imgp)
  im = np.array(im).astype(float)/255.
  means.append(np.mean(im))
  stds.append(np.std(im))
  if cnt > 10000:
    break
print("mean:",np.mean(means), ". variance: ",np.mean(stds))
                             
                        
                        
                                                                    
    
    
