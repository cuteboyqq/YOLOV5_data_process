#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 17:58:05 2022

@author: ali
"""

import shutil
import os
import glob
import tqdm

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def img2label_path(img_path,folder_name='labels',f_name=''):
    sa, sb = os.sep + 'images' + os.sep, os.sep + folder_name + os.sep
    return sb.join(img_path.rsplit(sa,1)).rsplit('.')[0] + f_name + '.txt'


def Analysis_path(label_path):
    file  = label_path.split(os.sep)[-1]
    filename = file.split(".")[0]
    return file,filename

def Get_filtered_dataset_by_label(img_dir,label_list,save_dir,for_labelImg):
    
    img_path_list = glob.glob(os.path.join(img_dir,"*.jpg"))
    c = 0
    PREFIX = colorstr('find wanted label datasets :')
    pbar = tqdm.tqdm(img_path_list,desc=f'{PREFIX}')
    for img_path in pbar:
        #print(c," ",img_path)
        
        
        label_path = img2label_path(img_path)
        #print(c," ",label_path)
        if os.path.exists(label_path):
            f_label = open(label_path,'r')
            lines = f_label.readlines()
            for line in lines:
                #print(line)
                label = line.split(" ")[0]
                #print(label)
                if int(label) in label_list:
                    #print("In label list~~~~")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                        
                    if for_labelImg:
                        save_labelImg_dir = os.path.join(save_dir,'for_labelImg')
                        if not os.path.exists(save_labelImg_dir):os.makedirs(save_labelImg_dir)
                        shutil.copy(img_path,save_labelImg_dir)
                        shutil.copy(label_path,save_labelImg_dir)
                        
                    save_img_dir = os.path.join(save_dir,'images')
                    if not os.path.exists(save_img_dir):os.makedirs(save_img_dir)
                    shutil.copy(img_path,save_img_dir)
                    save_label_dir = os.path.join(save_dir,'labels')
                    if not os.path.exists(save_label_dir):os.makedirs(save_label_dir)
                    shutil.copy(label_path,save_label_dir)
                    c+=1
    print("find ",c," datas")
    
    
def Get_wanted_labels_txt(wanted_label_list,img_dir,save_dir):
     img_path_list = glob.glob(os.path.join(img_dir,"*.jpg"))
     PREFIX = colorstr('find wanted label datasets :')
     pbar = tqdm.tqdm(img_path_list,desc=f'{PREFIX}')
     for img_path in pbar:
         label_path = img2label_path(img_path)
         if os.path.exists(label_path):
             
             file,filename = Analysis_path(label_path)
             
             f_label = open(label_path,'r')
             lines = f_label.readlines()
             for line in lines:
                 label = line.split(" ")[0]
                 #print(label)
                 if int(label) in wanted_label_list:
                     if not os.path.exists(save_dir):
                         os.makedirs(save_dir)
                     new_label_path = os.path.join(save_dir,file)
            
                     new_f_label = open(new_label_path,'a')
                     new_f_label.write(line)
                     new_f_label.close()
            
                 #print("matcch label")
                
if __name__=="__main__":
    '''
    names: ['person', 'bicycle', 'car', 'motorcycle', 'red ts', 'bus', 'green ts', 'truck', 'yellow ts', 'off ts',
            'red left ts', 'stop sign', 'green straight ts', 'green right ts', 'red right ts', 'green left ts', 'rider']
    '''
    img_dir = r"/home/ali/datasets/LISA_Youtube_COCO_BSTLD_NoCopy_11/train/images"
    #img_dir = r"/home/ali/datasets/LISA_NewYork_COCO_BSTLD_NoCopy_11/train/images"
    label_list = [8,10,12,13,14,15,17,18]
    #WPI datasets filter R,G,OFF,Yellow
    #label_list = [0,1,2,3,5,7,10,11,12,13,14,15,16]
    save_dir = r"/home/ali/datasets/LISA_Youtube_COCO_BSTLD_NoCopy_11/train/include_direction_TLs"
    for_labelImg = True
    Get_filtered_dataset_by_label(img_dir,label_list,save_dir,for_labelImg)
    
    #img_dir = r"/home/ali/datasets/WPI/for_train/images"
    #wanted_label_list = [0,1,2,3,5,7,8,10,12,13,14,15]
    #save_dir = r"/home/ali/datasets/WPI/for_train/final_label"
    #Get_wanted_labels_txt(wanted_label_list,img_dir,save_dir)