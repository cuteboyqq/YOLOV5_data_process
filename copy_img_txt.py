# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 21:59:00 2021

@author: User
"""

import os
import glob
import shutil


linux = True
window = False
def Analysis_path(path):
    if linux:
        file = path.split("/")[-1]
    else:
        file = path.split("\\")[-1]
    file_name =   file.split(".")[0]
    return file,file_name

copy_img = True
image_dir = '/home/ali/YOLOV5/assets/NewYork_drive/NewYork_train2_imgs'
save_img_dir = '/home/ali/datasets/NewYork_data/NewYork_train2_img_copy'
if copy_img:
    path_list = glob.iglob(os.path.join(image_dir,"*.jpg"))
else:
    path_list = glob.iglob(os.path.join(image_dir,"*.txt"))
cnt = 1
copy_times = 20

if not os.path.exists(save_img_dir):
    os.makedirs(save_img_dir)
for path in path_list:
    
    print(cnt," :",path)
    cnt+=1
    file,file_name = Analysis_path(path)
    print(file," ",file_name)
    if not file_name=="classes":
        for i in range(copy_times):
            new_file_name = ""
            new_file_name = file_name + "_" + str(i)
            if copy_img:        
                new_file = new_file_name + ".jpg"
            else:
                new_file = new_file_name + ".txt"
            print("new_file = ",new_file)
            copy_file_path = os.path.join(save_img_dir,new_file)
            print("path = ",path)
            print("copy_file_path = ",copy_file_path)
            shutil.copy(path,copy_file_path)
        
    
