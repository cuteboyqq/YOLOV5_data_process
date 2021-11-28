# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 21:59:00 2021

@author: User
"""

import os
import glob
import shutil



def Analysis_path(path):
    file = path.split("\\")[-1]
    file_name =   file.split(".")[0]
    return file,file_name


image_dir = r"D:\datasets-old\TLR_image"
save_img_dir = r"D:\datasets-old\TLR_txt_copy"
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
    for i in range(copy_times):
        new_file_name = ""
        new_file_name = file_name + "_" + str(i)
        new_file = new_file_name + ".txt"
        print(new_file)
        copy_file_path = os.path.join(save_img_dir,new_file)
        shutil.copy(path,copy_file_path)
        
    
