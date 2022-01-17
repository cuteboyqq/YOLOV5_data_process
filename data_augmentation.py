# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 18:06:15 2022

@author: admin
"""

import os
import shutil
import numpy as np
import glob
from pathlib import Path
import cv2



def load_image(img_path,img_size):
    img = cv2.imread(img_path) #BGR
    assert img is not None, 'Image Not Found' + img_path
    h0,w0 = img.shape[:2]
    r = img_size / max(h0,w0) #resize image to image_size
    if r!=1: # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r<1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0*r),int(h0*r)), interpolation=interp)
    return img, (h0,w0), img.shape[:2] #img, hw_original, hw_resized        

def img2label_paths(img_paths):
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep
    return [ sb.join(x.rsplit(sa,1)).rsplit('.')[0] + '.txt' for x in img_paths]



def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1,1,3) * [hgain, sgain, vgain] + 1 #random gains
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype # uint8
    
    x = np.arange(0, 256, dtype=np.int16)
    lut_h = ((x * r[0]) % 180).astype(dtype)
    lut_s = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_v = np.clip(x * r[2], 0, 255).astype(dtype)
    
    img_hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2_COLOR_HSV2BGR, dst=img) #no return needed
    

def Data_Augmentation(path,
                      img_size):
    prefix=''
    f = []  # image files
    IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
    
    print("Start Analysis Train images...")
    print("Train image input size = ",img_size)
    for p in path if isinstance(path, list) else [path]:
        p = Path(p)  # os-agnostic
        print(p)
        
        if p.is_dir():  # dir
            f += glob.glob(str(p / '**' / '*.*'), recursive=True)
            #for file in f:
                #print(c,":",file)
                #c+=1
            
            # f = list(p.rglob('*.*'))  # pathlib
        elif p.is_file():  # file
            with open(p) as t:
                t = t.read().strip().splitlines()
                parent = str(p.parent) + os.sep
                f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
        else:
            raise Exception(f'{prefix}{p} does not exist')
            
    img_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
    img_size = 320
    for img_f in img_files:
        print(img_f)
        
        img, _, (h,w) = load_image(img_f,img_size)
        print(_," (",h,",",w,")")
        
    label_files = img2label_paths(img_files)
    
    #for label_f in label_files:
        #print(label_f)
    
        




if __name__=="__main__":
    path =  "/home/ali/datasets-smallTLR/coco/images/train2017"
    img_size = 320
    Data_Augmentation(path,
                      img_size)
    