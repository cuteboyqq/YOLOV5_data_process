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

def Get_filtered_dataset_by_label(img_dir,label_list,save_dir):
    
    img_path_list = glob.glob(os.path.join(img_dir,"*.jpg"))
    c = 0
    PREFIX = colorstr('find wanted label datasets :')
    pbar = tqdm.tqdm(img_path_list,desc=f'{PREFIX}')
    for img_path in pbar:
        #print(c," ",img_path)
        
        
        label_path = img2label_path(img_path)
        #print(c," ",label_path)
        
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
                shutil.copy(img_path,save_dir)
                shutil.copy(label_path,save_dir)
                c+=1
    print("find ",c," datas")
                
if __name__=="__main__":
    img_dir = r"/home/ali/datasets/bdd100k-TLs/train/labeled/images"
    label_list = [10,12,13,14,15]
    save_dir = r"/home/ali/datasets/bdd100k-TLs/train/labeled/TL-directions-for-labelImg"
    Get_filtered_dataset_by_label(img_dir,label_list,save_dir)
    