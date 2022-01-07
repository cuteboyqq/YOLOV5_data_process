#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 15:39:01 2022

@author: ali
"""

import os
import glob
import tqdm
import shutil

label_name = ['person', 'bicycle', 'car', 'motorcycle', 'red ts', 'bus', 'green ts', 'truck', 'yellow ts', 'off ts',
        'red left ts', 'stop sign', 'green straight ts', 'green right ts', 'red right ts', 'green left ts', 'rider']

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
def Analysis_path(yolov5_infer_txt_path):
    file = yolov5_infer_txt_path.split("/")[-1]
    file_dir = os.path.dirname(yolov5_infer_txt_path)
    
    return file,file_dir


def Add_TL_Label_To_BDD100K(bdd100k_txt_dir,
                            yolov5_infer_label_dir,
                            save_new_label_dir,
                            only_save_new_label_with_TL_dr,
                            wanted_label_list):
    
    if not os.path.exists(save_new_label_dir):
        os.makedirs(save_new_label_dir)
    yolov5_infer_label_txt_list = glob.glob(os.path.join(yolov5_infer_label_dir,'*.txt'));
    c = 1
    PREFIX = colorstr('Match YoloV5 infer label.txt with GT:')
    pbar = tqdm.tqdm(yolov5_infer_label_txt_list,desc=f'{PREFIX}')  # progress bar
    for _ in pbar:
    #PREFIX = colorstr('Match YoloV5 infer label.txt with GT: ')
    #for yolov5_infer_txt_path in tqdm.tqdm(yolov5_infer_label_txt_list,desc=f'{PREFIX}'):
        yolov5_infer_txt_path = _
        #pbar.desc = f'{yolov5_infer_txt_file}'
        #print(c," ",yolov5_infer_txt_file)
        file,file_dir = Analysis_path(yolov5_infer_txt_path)
        
        bdd100k_txt_path = os.path.join(bdd100k_txt_dir,file)
        #print(c," ",bdd100k_txt_path)
        
        
        if not os.path.exists(bdd100k_txt_path):
            #print("not exists")
            continue
        else:
            shutil.copy(bdd100k_txt_path,save_new_label_dir)
            final_bdd100k_txt_path = os.path.join(save_new_label_dir,file)
            f_yolo = open(yolov5_infer_txt_path,'r')
            lines = f_yolo.readlines()
            #print(lines)
            add=False
            for line in lines:
                #print(line)
                label =  line.split(" ")[0]
                #print("label = ",str(label_name[int(label)]))
                if int(label) in wanted_label_list:
                    add = True
                    #print("find ",label)
                    #PREFIX = colorstr('')
                    #pbar.set_description(f'{file} add {label}')
                    f_bdd100k = open(final_bdd100k_txt_path,'a+')
                    #f_bdd100k.write("\n")
                    f_bdd100k.write(line)
                    f_bdd100k.close()
            co = str(c)
            PREFIX = colorstr('Find YoloV5 infer result of label.txt with TL labels :')
            pbar.desc = f'{PREFIX} Add TL label to BDD100K label.txt count : {co}'
            if add is True:
                if not os.path.exists(only_save_new_label_with_TL_dr):
                    os.makedirs(only_save_new_label_with_TL_dr)
                shutil.copy(final_bdd100k_txt_path,only_save_new_label_with_TL_dr)
                c+=1
            #for line in lines:
                #print(line,end='')
            f_yolo.close()
            
if __name__=="__main__":
    wanted_label_list = [4,6,8,9,10,11,12,13,14,15]
    bdd100k_txt_dir = "/home/ali/bdd100k_real_ori/labels/train_for_test"
    yolov5_infer_label_dir = "/home/ali/YOLOV5/runs/detect/bdd100k-train/labels"
    save_new_label_dir = "/home/ali/bdd100k_real_ori/labels/train_for_test_add_TL"
    only_save_new_label_with_TL_dr = "/home/ali/bdd100k_real_ori/labels/train_for_test_add_TL_only"
    Add_TL_Label_To_BDD100K(bdd100k_txt_dir,
                            yolov5_infer_label_dir,
                            save_new_label_dir,
                            only_save_new_label_with_TL_dr,
                            wanted_label_list)           