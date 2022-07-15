#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 17:45:48 2022

@author: ali
"""

import glob
import os



def img2label_paths(img_paths):
    sa = os.sep + 'images' + os.sep
    sb = os.sep + 'labels' + os.sep    
    return [ sb.join(x.rsplit(sa)).rsplit('.')[0]+ '.txt' for x in img_paths] 

def img2label_path(img_path):
    sa = os.sep + 'images' + os.sep
    sb = os.sep + 'labels' + os.sep    
    return sb.join(img_path.rsplit(sa)).rsplit('.')[0]+ '.txt'

TRAIN=True
VAL=False

if TRAIN:
    img_dir = "/home/ali/datasets/factory_data/images/train"
    label_dir = "/home/ali/datasets/factory_data/labels/train"
elif VAL:
    img_dir = "/home/ali/datasets/factory_data/images/val"
    label_dir = "/home/ali/datasets/factory_data/labels/val"
    

label_path_list = glob.iglob(os.path.join(label_dir,'*.txt'))
img_path_list = glob.iglob(os.path.join(img_dir,'*.jpg'))

#for im_p in img_path_list:
    #print(im_p)



label_path_list = img2label_paths(img_path_list)

if TRAIN:
    save_txt_path = "/home/ali/datasets/factory_data/factory_data.txt"
elif VAL:
    save_txt_path = "/home/ali/datasets/factory_data/factory_data_val.txt"

#if not os.path.exists(save_txt_path):
    #os.makedirs(save_txt_path)
import cv2
line=[]
with open(save_txt_path,'w') as final_f:
    #final_f.write("Create a new file")
    img_path_list = glob.iglob(os.path.join(img_dir,'*.jpg'))
    for img_path in img_path_list:
        #line = []
        print(img_path)
        img = cv2.imread(img_path)
        h,w,c = img.shape
        label_path = img2label_path(img_path)
        if os.path.exists(label_path):
            line.append(img_path)
            line.append(' ')
        
        print(label_path)
        if os.path.exists(label_path):
            f = open(label_path,'r')
            lines = f.readlines()
            for l in lines:
                
                l_list = l.split(" ")
                print('l_list[0] = {}'.format(l_list[0]))
                new_l_list = [0,0,0,0,0]
                #convert lxywh into x1,y1,x2,y2,l
                new_l_list[0] = str( int(float(l_list[1])*w) )
                new_l_list[1] = str( int(float(l_list[2])*h) )
                new_l_list[2] = str( int(float(l_list[1])*w + (float(l_list[3])*w/2.0)) ) 
                new_l_list[3] = str( int(float(l_list[2])*h + (float(l_list[4])*h/2.0)) )
                print('2. l_list[0] = {}'.format(l_list[0]))
                new_l_list[4] = l_list[0]
                
                new_line = ",".join(new_l_list)
                print("new_line =", new_line)
                line.append(new_line)
                line.append(' ')
                #ll = l.split("\n")[0]
                #print(ll)
                #line.append(ll)
                #line.append(' ')
            line.append('\n')
    final_f.writelines(line)    
        #line_list = line.split(" ")
        #line_data = ''
        #for data in line_list:
            #line_data
    
    
final_f.close()
        
            
                
            


#for img_path in img_path_list:
    #print(img_path)
        
