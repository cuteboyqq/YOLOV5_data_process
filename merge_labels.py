#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 10:35:27 2022

@author: ali
"""

import shutil
import os
import numpy as np
import glob
import tqdm

class_name = ['person', 'bicycle', 'car', 'motorcycle', 'red ts', 'bus', 'green ts', 'truck', 'yellow ts', 'off ts',
        'red left ts', 'stop sign', 'green straight ts', 'green right ts', 'red right ts', 'green left ts', 'rider']

new_class_name = ['person', 'motor', 'car', 'red ts', 'green ts', 'yellow ts', 'off ts', 'stop sign', 'other ts']


counter = {}
for c in new_class_name:
    counter[c] = 0

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


def Analysis_path(bdd100k_label_txt_path):
    file = bdd100k_label_txt_path.split("/")[-1]
    file_dir = os.path.dirname(bdd100k_label_txt_path)
    return file,file_dir

def Merge_BDD100K_labels(bdd100k_label_dir,
                         ori_class,
                         new_class,
                         save_new_bdd100k_label_dir,
                         view_log):
    
    if not os.path.exists(save_new_bdd100k_label_dir):
        os.makedirs(save_new_bdd100k_label_dir)
    
    c= 1
    ''' get bdd100k all label.txt paths, and put all paths in list '''
    original_bdd100k_label_list = glob.glob(os.path.join(bdd100k_label_dir,"*.txt"))

    #for label_txt_path in original_bdd100k_label_list:
        #print(c," : ",label_txt_path)
        #c+=1
    ''' use tqdm.tqdm to show percentage of progress '''
    PREFIX = colorstr('Ana;ysis labels.txt :')
    pbar = tqdm.tqdm(original_bdd100k_label_list,desc=f'{PREFIX}')
    ''' use for loop to go through all label.txt path '''   
    for bdd100k_label_txt_path in pbar:
        
        file,file_dir = Analysis_path(bdd100k_label_txt_path)
        
        f_bdd100k_ori = open(bdd100k_label_txt_path,'r')
        lines = f_bdd100k_ori.readlines()
        
        ''' the new label.txt path '''
        new_label_txt_path = os.path.join(save_new_bdd100k_label_dir,file)
        '''create and open new label.txt and ready to write newl label x y w h  to new label.txt'''
        new_label_txt = open(new_label_txt_path,'a+')
        
        ''' start analysis original label.txt'''
        for line in lines:
            #print(line)
            ''' extract label list'''
            label_list = line.split(" ")[0:1]
            ''' extract label string'''
            label = line.split(" ")[0]
            
            #x = line.split(" ")[1]
            #y = line.split(" ")[2]
            #w = line.split(" ")[3]
            #h = line.split(" ")[4]
            ''' extract xywh list '''
            xywh_list = line.split(" ")[1:]
            if view_log:
                print("ori:",label_list)
                print(xywh_list)
            
            
            class_index = ori_class.index(int(label))
            ''' get new label'''
            new_label = new_class[class_index]
            ''' count number of new label '''
            new_label_name = new_class_name[int(new_label)]
            counter[new_label_name]+=1
            '''new label list'''
            new_label_list = [str(new_label)]
            '''concat new label list and xywh list'''
            new_labelxywh_list = new_label_list + xywh_list
            if view_log:
                print("new : ",new_labelxywh_list)
            
            '''extract new labelxywh list into string labelxywh''' 
            new_labelxywh = " ".join(new_labelxywh_list)
            if view_log:
                print("new : ",new_labelxywh)
            
            ''' write newl label x y w h  to new label.txt'''
            new_label_txt.write(new_labelxywh)
            
        '''end of analysis original label.txt, then close new label.txt'''    
        new_label_txt.close()
            
            #print(class_name[int(label)]," ",new_class_name[new_label])
            #for i in range(1000):
                #print(class_name[int(label)]," ",new_class_name[new_label])
            #new_line = ' '.join(new_label,x,y,w,h)
            #print(new_line)
        #print(c," : ",label_txt_path)
        #c+=1
    
    print(counter)
    
def get_args():
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-l','--labels',help="labels dir",type=str, default='/home/ali/bdd100k_merge_label/labels/')
    parser.add_argument('-t','--type',help="type of datasets", choices=['train','val'],default='val')
    parser.add_argument('--log','--view-log',action='store_true',help="show detail of merge labels")
    return parser.parse_args()    


if __name__=="__main__":
    ''' get parameter settings from console '''   
    args = get_args()
    ''' analysis parameter settings '''    
    if args.type=='train':
        bdd100k_label_dir = os.path.join(args.labels,'train')
        save_new_bdd100k_label_dir = os.path.join(args.labels,'train-new')
    else:
        bdd100k_label_dir = os.path.join(args.labels,'val')
        save_new_bdd100k_label_dir = os.path.join(args.labels,'val-new')
        
    if args.log ==True:
        view_log = True
    else:
        view_log = False
    
    
    
    #bdd100k_label_dir = '/home/ali/bdd100k_merge_label/labels/val_test'
    class_name = ['person', 'bicycle', 'car', 'motorcycle', 'red ts', 'bus', 'green ts', 'truck', 'yellow ts', 'off ts',
            'red left ts', 'stop sign', 'green straight ts', 'green right ts', 'red right ts', 'green left ts', 'rider']
    new_class_name = ['person', 'motor', 'car', 'red ts', 'green ts', 'yellow ts', 'off ts', 'stop sign', 'other ts']
    #save_new_bdd100k_label_dir = '/home/ali/bdd100k_merge_label/labels/val_test_new'
    ori_class = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    new_class = [0,1,2,1,3,2,4,2,5,6, 8, 7, 8, 8, 8, 8, 1]
    Merge_BDD100K_labels(bdd100k_label_dir,
                         ori_class,
                         new_class,
                         save_new_bdd100k_label_dir,
                         view_log)