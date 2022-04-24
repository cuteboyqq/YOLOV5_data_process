# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 21:29:07 2021

@author: User
"""
import os
import glob
import cv2
import shutil



def Analysis_label_txt_path(label_txt_path):
    txt_file = label_txt_path.split("/")[-1]
    txt_name = txt_file.split(".")[0]
    return txt_file,txt_name
'''
============================================================================================
def Filter_coco_class(train_label_dir,
                      save_new_train_label_dir,
                      train_image_dir,
                      save_new_train_image_dir,
                      label_list)
The function is able to get the dataset including the wanted label, so the unwanted label will be filter out
=============================================================================================================
'''
def Filter_coco_class(train_label_dir,
                      save_new_train_label_dir,
                      train_image_dir,
                      save_new_train_image_dir,
                      label_list):
    print("Alister Test !")
    train_label_dir_list = glob.iglob(os.path.join(train_label_dir,'*.txt'))
    save_count = 0
    for label_txt_path in train_label_dir_list:
        #print(label_txt_path)
       
        txt_file,txt_name = Analysis_label_txt_path(label_txt_path)
        #print(txt_file)
       
        save_new_txt_file_path = os.path.join(save_new_train_label_dir,txt_file)
        if not os.path.exists(save_new_train_label_dir):
            os.makedirs(save_new_train_label_dir)
        #if save_count <= 5000:
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
                print(img_file)
                save_new_train_img_path = os.path.join(save_new_train_image_dir,img_file)
                train_image_path = os.path.join(train_image_dir,img_file)
                if not os.path.exists(save_new_train_image_dir):
                    os.makedirs(save_new_train_image_dir)
                if os.path.exists(train_image_path):
                    #if  save_count <= 5000:
                    shutil.copy(train_image_path,save_new_train_image_dir)
                    save_count+=1
                    print(save_count,' save image :',img_file)
'''
===========================================================================================

def Filter_data_by_class(input_img_dir, 
                         output_img_dir,
                         input_label_dir,
                         output_label_dir,
                         wanted_label_list):

Note: This funtion  is able to  get datasset including directional TS by giving the list of directional TS labels 

=============================================================================================
'''

def Filter_data_by_class(input_img_dir, 
                         output_img_dir,
                         input_label_dir,
                         output_label_dir,
                         wanted_label_list):
    input_label_path_list = glob.iglob(os.path.join(input_label_dir,'*.txt'))
    c = 1
    for label_txt_path in input_label_path_list:
        #print(label_txt_path)
        #print(c)
        #c+=1
        txt_file,txt_name = Analysis_label_txt_path(label_txt_path)
        save_new_txt_file_path = os.path.join(output_label_dir,txt_file)
        copy_new_file = False
        with open(label_txt_path) as f:
            lines = f.readlines()
            for line in lines:
                #print(line)
                label = line.split(" ")[0]
                for want_label in wanted_label_list:
                    #print('want_label = ' ,want_label)
                    #print('label = ',label)
                    if label == str(want_label):
                        copy_new_file = True
        if copy_new_file==True:
            img = txt_name + '.jpg'
            input_img_path = os.path.join(input_img_dir,img)
            if not os.path.exists(output_img_dir):
                os.makedirs(output_img_dir)
            shutil.copy(input_img_path,output_img_dir)
            if not os.path.exists(output_label_dir):
                os.makedirs(output_label_dir)
            shutil.copy(label_txt_path,output_label_dir)
            #print('copy :',end='\r ')
            strq = "copy :" + str(c)
            print("\r",strq,end="",flush=True)
            c+=1
                            
        
        
if __name__=="__main__":
    FILTER_LABEL = True
    if FILTER_LABEL==True:
        train_label_dir = '/home/ali/datasets/coco/labels/train2017-ori'
        save_new_train_label_dir = '/home/ali/datasets/coco/labels/train_7class_NOTS'
        train_image_dir = '/home//ali/datasets/coco/images/train2017-ori'
        save_new_train_image_dir = '/home/ali/datasets/coco/images/train2017_7class_NOTS'
        label_list = [0,1,2,3,5,7,11]
        
        
        Filter_coco_class(train_label_dir,
                              save_new_train_label_dir,
                              train_image_dir,
                              save_new_train_image_dir,
                              label_list)
    #=======================================================================================
    FILTER_DATA_BY_CLASS = False
    if FILTER_DATA_BY_CLASS==True:
        input_img_dir = '/home/ali/datasets/coco/images/val2017'
        output_img_dir = '/home/ali/datasets/coco/images/val2017_TS'
        input_label_dir = '/home/ali/datasets/coco/labels/val2017'
        output_label_dir = '/home/ali/datasets/coco/labels/val_TS'
        wanted_label_list = [9]
        
        
        Filter_data_by_class(input_img_dir, 
                                 output_img_dir,
                                 input_label_dir,
                                 output_label_dir,
                                 wanted_label_list)
    
        
            
            
        
        
        
