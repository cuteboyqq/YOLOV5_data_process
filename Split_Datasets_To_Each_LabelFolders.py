#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 12:17:15 2022

@author: ali
"""

import os
import shutil
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


def Analysis_dir(img_dir):
    img_dir_dir = os.path.dirname(img_dir)
    folder_name = img_dir.split(os.sep)[-1]
    return folder_name,img_dir_dir

def img2label_path(img_path,folder_name='labels',f_name=''):
    sa, sb = os.sep + 'images' + os.sep, os.sep + folder_name + os.sep
    return sb.join(img_path.rsplit(sa,1)).rsplit('.')[0] + f_name + '.txt'

def Split_Datasets_labels_To_Each_Folders(img_dir,class_name,class_txt):
    
    img_path_list = glob.glob(os.path.join(img_dir,"*.jpg"))
    
    PREFIX = colorstr('Start Split Datasets by labels')
    pbar = tqdm.tqdm(img_path_list,desc=f'{PREFIX}')
    c=1
    
    folder_name,img_dir_dir = Analysis_dir(img_dir)
    save_dir = os.path.join(img_dir_dir,"split_datasets_by_labels") 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for img_path in pbar:
        label_path = img2label_path(img_path)
        #print(label_path)
        if os.path.exists(label_path):
            f_label = open(label_path,'r')
            lines = f_label.readlines()
            
            for line in lines:
                label = line.split(" ")[0]
                #print(label)
                folder_name = class_name[int(label)]
                save_folder_dir = os.path.join(save_dir,folder_name)
                if not os.path.exists(save_folder_dir):
                    os.makedirs(save_folder_dir)
                
                if not os.path.exists(os.path.join(save_folder_dir,'classes.txt')):
                    shutil.copy(class_txt,save_folder_dir)
                shutil.copy(label_path,save_folder_dir)
                shutil.copy(img_path,save_folder_dir) 
                
                #print(folder_name)
                
                     
        c+=1
        
    print(c)
        
        
if __name__=="__main__":
    class_txt = r"/home/ali/datasets/train_video/classes.txt"
    img_dir = r"/home/ali/datasets/train_video/11-all/images"
    class_name = ['person', 'bicycle', 'car', 'motorcycle', 'red ts', 'bus', 'green ts', 'truck', 'yellow ts', 'off ts',
            'red left ts', 'stop sign', 'green straight ts', 'green right ts', 'red right ts', 'green left ts', 'rider','yellow left ts','yellow right ts']
    Split_Datasets_labels_To_Each_Folders(img_dir,class_name,class_txt)