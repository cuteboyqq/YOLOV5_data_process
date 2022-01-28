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

boxes = np.array(boxes)

n = len(boxes)
print("num of data in boxes : \n",n)
anchors = np.array(boxes)[np.random.choice(n, 6, replace=True)]
print("initial random anchors (w,h) (get by boxes of random) = \n",anchors)
#anchors_ = boxes[2]
#print(anchors_)
labels_ = np.zeros((n,))

print("initail labels : \n",labels_)


n_one = np.array(boxes)[:, 0,np.newaxis]
one_k = anchors[np.newaxis,:, 0]
print("expend box w dimension to n x 1 : \n",n_one)
print("expend anchor w dimension to 1 x k : \n",one_k)


print("===========================")
#w_min_2 = np.minimum(n_one, one_k)
#print(w_min_2)
print("==============================")
w_min = np.minimum( boxes[:, 0,np.newaxis], anchors[np.newaxis,:, 0] )
print("w_min = min(boxes_w,anchors_w) \n",  w_min)
print("=================================")
h_min = np.minimum( boxes[:, 1,np.newaxis], anchors[np.newaxis,:, 1] )
print("h_min = min(boxs_h,anchors_h) \n", h_min)

inter = w_min * h_min
print("=============================")
print("inter = w_min * h_min \n",inter)

#calculate the union
box_area = boxes[:,0] * boxes[:,1]
print("===============================")
print("box_area = box[:,0] * box[:,1]\n",box_area)
anchor_area = anchors[:,0] * anchors[:,1]
print("=====================================")
print("anchor_area = anchors[:,0] * anchors[:,1]\n",anchor_area)

box_area_expend_dim = box_area[:,np.newaxis]
print("=====================================")
print("box_area_expend_dim = box_area[:,np.newaxis]\n",box_area_expend_dim)

anchor_area_expend_dim = anchor_area[np.newaxis]
print("================================")
print("anchor_area_expend_dim = anchor_area[np.newaxis]\n",anchor_area_expend_dim)


union = box_area[:,np.newaxis] + anchor_area[np.newaxis]
print("union = box_area[:,np.newaxis] + anchor_area[np.newaxis]\n",union)
iou = inter / (union - inter)
down = union - inter
print("==========================================")
print("down = union - inter \n",down)
print("=========================================")
print("iou = inter / (union - inter)\n",iou)

distance = 1 - iou
print("distance =  1 - iou \n",distance)

curlabel = np.argmin(distance,axis=1)
print("===============================================")
print("curlabel = np.argmin(distance,axis=1)\n",curlabel)



boxes_0 = boxes[curlabel==0]
print("=====================================")
print("boxes_0 = boxes[curlabel==0]\n",boxes_0)

anchor_0 = np.mean(boxes[curlabel==0],axis=0)
print("==========================================")
print(" anchor_0 = np.mean(boxes[curlabel==0],axis=0) \n",anchor_0)

#anchors_ = None
print("================================================")
print("anchors :")
for i in range(6):
    anchors[i] = np.mean(boxes[curlabel==i],axis=0)
    print(anchors[i])    


