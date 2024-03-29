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
    img_dir = "/home/ali/factory_video/images/train"
    label_dir = "/home/ali/factory_video/labels/train"
elif VAL:
    img_dir = "/home/ali/factory_video/images/val"
    label_dir = "/home/ali/factory_video/labels/val"
    

label_path_list = glob.iglob(os.path.join(label_dir,'*.txt'))
img_path_list = glob.iglob(os.path.join(img_dir,'*.jpg'))

#for im_p in img_path_list:
    #print(im_p)



label_path_list = img2label_paths(img_path_list)

if TRAIN:
    save_txt_path = "/home/ali/factory_video/images/factory_data_noaug_20220728.txt"
elif VAL:
    save_txt_path = "/home/ali/factory_video/images/factory_data_val_blur9_20220728.txt"

#if not os.path.exists(save_txt_path):
    #os.makedirs(save_txt_path)
import cv2
line=[]
X1Y1X2Y2 = True
if not X1Y1X2Y2:
    Y1X1Y2X2 = True
else:
    Y1X1Y2X2 = False
    
with open(save_txt_path,'w') as final_f:
    #final_f.write("Create a new file")
    img_path_list = glob.iglob(os.path.join(img_dir,'*.jpg'))
    for img_path in img_path_list:
        #line = []
        print(img_path)
        img = cv2.imread(img_path)
        h,w,c = img.shape
        print('h: {}, w:{}, c:{}'.format(h,w,c))
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
                
                if X1Y1X2Y2:
                    #convert (l,x_center,y_center,w,h) into (x_topleft, y1_topleft, x2_downright, y2_downright,l)
                    new_l_list[0] = str( int( float(l_list[1])*w - float(l_list[3])*w/2.0 ) ) #x_topleft = x_center - w/2.0
                    new_l_list[1] = str( int( float(l_list[2])*h - float(l_list[4])*h/2.0 ) ) #y1_topleft = y_center - h/2.0
                    new_l_list[2] = str( int( float(l_list[1])*w + float(l_list[3])*w/2.0 ) ) #x2_downright = x_center + w/2.0
                    new_l_list[3] = str( int( float(l_list[2])*h + float(l_list[4])*h/2.0 ) ) #y2_downright = y_center + h/2.0
                    print('2. l_list[0] = {}'.format(l_list[0]))
                    
                elif Y1X1Y2X2:
                    #convert lxywh into y1,x1,y2,x2,l
                    new_l_list[1] = str( int( float(l_list[1])*w - float(l_list[3])*w/2.0 ) ) #x_topleft = x_center - w/2.0
                    new_l_list[0] = str( int( float(l_list[2])*h - float(l_list[4])*h/2.0 ) ) #y1_topleft = y_center - h/2.0
                    new_l_list[3] = str( int( float(l_list[1])*w + float(l_list[3])*w/2.0 ) ) #x2_downright = x_center + w/2.0
                    new_l_list[2] = str( int( float(l_list[2])*h + float(l_list[4])*h/2.0 ) ) #y2_downright = y_center + h/2.0
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
        
