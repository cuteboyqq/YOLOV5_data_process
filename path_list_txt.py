# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 18:54:26 2021

@author: User
"""
import os
import glob

search_dir = r"/home/ali/datasets/bdd100k/images/val" 
path_list_txt = r"/home/ali/datasets/bdd100k/val.txt"

path_list = glob.iglob(os.path.join(search_dir,"*.jpg"))
path_list2 = glob.iglob(os.path.join(search_dir,"*.png"))
with open(path_list_txt,'w') as f:
    
    for path in path_list:
        print(path)
        f.write(path)
        f.write("\n")
    
    for path2 in path_list2:
        print(path2)
        f.write(path2)
        f.write("\n")
    

f.close()
    