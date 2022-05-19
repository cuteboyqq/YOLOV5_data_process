#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 13:34:46 2022

@author: ali
"""
import cv2
import glob
import os
import shutil

def Analysis_path(img_path):
    img_name = img_path.split(os.sep)[-1]
    return img_name
 


def Resize_Images(img_size,img_dir,save_dir):
    img_path_list = glob.iglob(os.path.join(img_dir,'*.png'))
    
    for img_path in img_path_list:
        print(img_path)
        
        img_name = Analysis_path(img_path)
        
        img = cv2.imread(img_path)
        img_resize = cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_AREA)
        
        cv2.imwrite(os.path.join(save_dir,img_name) , img_resize)
        

if __name__=="__main__":
    img_size = 320
    img_dir = '/home/ali/YOLOV4/assets/images'
    save_dir = '/home/ali/projects/bub_cv25_adk2.3.0/rtos/refapp/cnn_testbed/cv/public_nn/images/yolov4/dra'
    Resize_Images(img_size,img_dir,save_dir)
    
    
