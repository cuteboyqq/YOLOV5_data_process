# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 18:54:26 2021

@author: User
"""
import os
import glob

search_dir = r"D:\datasets-smallTLR\coco\images\val2017" 
path_list_txt = r"D:\datasets-smallTLR\coco\val2017_new.txt"

path_list = glob.iglob(os.path.join(search_dir,"*.jpg"))

with open(path_list_txt,'w') as f:
    
    for path in path_list:
        print(path)
        f.write(path)
        f.write("\n")
    

f.close()
    