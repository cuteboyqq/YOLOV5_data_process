#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 10:15:11 2022

@author: ali
"""

import os
import glob
import matplotlib.pyplot as plt
import tqdm

def Analysis_path(label_path):
    file = label_path.split(os.sep)[-1]
    return file


def draw_dict(D):
    plt.bar(*zip(*D.items()))
    #=====================================================
    #plt.bar(range(len(D)),list(D.values()), align='center')
    #plt.xticks(range(len(D)), list(D.keys()))
    
    #==========================================================
    #names = list(D.keys())
    #values = list(D.values())

    #tick_label does the some work as plt.xticks()
    #plt.bar(range(len(D)),values,tick_label=names)
    plt.savefig('bar.png')
    plt.show()

def Count_labels(label_dir,names):
    label_dict = {}
    label_count = [0]*19
    label_path_list = glob.glob(os.path.join(label_dir,"*.txt"))
    
    pbar = tqdm.tqdm(label_path_list)
    #print("--------------------------------")
    print(label_path_list)
    for label_path in pbar:
        #print(label_path)
        file = Analysis_path(label_path)
        if not (file=="classes.txt"):
            f_label = open(label_path,"r")
            lines = f_label.readlines()
            for line in lines:
                label = line.split(" ")[0]
                label_count[int(label)] = label_count[int(label)] + 1
                    
        f_label.close()
    
    for i in range(19):
        label_dict[names[i]] = label_count[i]
    print(label_count)
    print(label_dict)
    draw_dict(label_dict)
    return label_count
        
if __name__=="__main__":
    names=['person', 'bicycle', 'car', 'motorcycle', 'red ts', 'bus', 'green ts', 'truck', 'yellow ts', 'off ts',
            'red left ts', 'stop sign', 'green straight ts', 'green right ts', 'red right ts', 'green left ts', 'rider','yellow left ts','yellow right ts']
    
    #label_dir = r"/home/ali/datasets/LISA_NewYork_COCO_BSTLD_NoCopy_11/train/labels"
    #label_dir = r"/home/ali/datasets/bdd100k/labels/train"
    label_dir = r"/home/ali/datasets/WPI/for_train/labels"
    #label_dir = r"/home/ali/datasets/bdd100k-TLs/train/train_TLs_labeled_2/labels"
    #label_dir = r"/home/ali/datasets/LISA_NewYork_COCO_BSTLD_NoCopy_11/train/include_dir_TLs/aug_flip_hsv_blur_Gaussian_5/labels"
    #label_dir = r"/home/ali/datasets/LISA_NewYork_COCO_BSTLD_NoCopy_11/train/include_dir_TLs/labels"
    #label_dir = r"/home/ali/datasets/bdd100k_small/labels/train"
    label_count = Count_labels(label_dir,names)