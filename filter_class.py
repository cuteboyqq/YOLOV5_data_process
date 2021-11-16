# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 21:29:07 2021

@author: User
"""
import os
import glob
import cv2
import shutil


train_label_dir = r'D:\datasets\coco\labels\train2017'
save_new_train_label_dir = r'D:\datasets\coco\labels\train2017_5class'
train_image_dir = r'D:\datasets\coco\images\train2017'
save_new_train_image_dir = r'D:\datasets\coco\images\train2017_5class'
label_list = [0,1,2,3,5,7,9,11]
def Analysis_label_txt_path(label_txt_path):
    txt_file = label_txt_path.split("\\")[-1]
    txt_name = txt_file.split(".")[0]
    return txt_file,txt_name


train_label_dir_list = glob.iglob(os.path.join(train_label_dir,'*.txt'))
save_count = 0
for label_txt_path in train_label_dir_list:
    #print(label_txt_path)
    txt_file,txt_name = Analysis_label_txt_path(label_txt_path)
    save_new_txt_file_path = os.path.join(save_new_train_label_dir,txt_file)
    if not os.path.exists(save_new_train_label_dir):
        os.makedirs(save_new_train_label_dir)
    with open(label_txt_path,'r') as f:
        add_new_file = False
        lines = f.readlines()
        for line in lines:
            #print(line)
            label = line.split(" ")[0]
            for want_label in label_list:
                #print('want_label = ' ,want_label)
                #print('label = ',label)
                if label == str(want_label):
                    #print('find wanted labels = ',want_label)
                    with open(save_new_txt_file_path,'a') as fw:
                        fw.writelines(line)
                        #print('write to new txtfile')
                        add_new_file = True
                        #cv2.wait()
                    fw.close()
        
        if add_new_file:
            img_file = txt_name + '.jpg'
            save_new_train_img_path = os.path.join(save_new_train_image_dir,img_file)
            train_image_path = os.path.join(train_image_dir,img_file)
            if not os.path.exists(save_new_train_image_dir):
                os.makedirs(save_new_train_image_dir)
            shutil.copy(train_image_path,save_new_train_image_dir)
            
            save_count+=1
            print(save_count,' save image :',img_file)
            
            
        
        
        