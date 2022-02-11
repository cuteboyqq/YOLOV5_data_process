#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 10:15:11 2022

@author: ali
"""

import os
import glob

def Analysis_path(label_path):
    file = label_path.split(os.sep)[-1]
    return file

def Count_labels(label_dir):
    label_count = [0]*17
    label_path_list = glob.glob(os.path.join(label_dir,"*.txt"))
    #print("--------------------------------")
    print(label_path_list)
    for label_path in label_path_list:
        print(label_path)
        file = Analysis_path(label_path)
        if not (file=="classes.txt"):
            f_label = open(label_path,"r")
            lines = f_label.readlines()
            for line in lines:
                label = line.split(" ")[0]
                label_count[int(label)] = label_count[int(label)] + 1
            
        f_label.close()
    
    
    print(label_count)
    return label_count
        
if __name__=="__main__":
    #label_dir = r"/home/ali/datasets/LISA_NewYork_Taipei_COCO_BSTLD_NoCopy_11/train/labels"
    #label_dir = r"/home/ali/datasets/bdd100k/labels/train"
    #label_dir = r"/home/ali/datasets/WPI/for_labelImg_wpi_no_off"
    label_dir = r"/home/ali/datasets/bdd100k-TLs/train/labeled/labels"
    label_count = Count_labels(label_dir)