# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 01:12:38 2022

@author: User
"""

import random
import numpy as np


boxes = [ [1,2],
         [10,20],
         [25,78],
         [100,150],
         [16,48],
         [48,35],
         [13,28],
         [3,4],
         [150,100],
         [123,456],
         [24,15],
         [18,16],
         [100,120],
         [15,45]]

#boxex = np.array(boxes)
n = len(boxes)
print(n)
anchors = np.array(boxes)[np.random.choice(n, 6, replace=True)]
print(anchors)
#anchors_ = boxes[2]

#print(anchors_)

labels_ = np.zeros((n,))

print(labels_)


n_one = np.array(boxes)[:, 0,np.newaxis]
one_k = anchors[np.newaxis,:, 0]
print(n_one)
print(one_k)


print("===========================")
w_min = np.minimum(n_one, one_k)
print(w_min)

