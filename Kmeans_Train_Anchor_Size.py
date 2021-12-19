# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 22:46:24 2021

@author: User
"""
import glob
import os
from pathlib import Path
import numpy as np
#import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans
from PIL import ExifTags, Image, ImageOps

stop_num = 75000
path = r"D:\datasets-smallTLR\coco\images\train2017"
prefix=''
f = []  # image files
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
c= 1
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of multiprocessing threads

def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

print("Start Analysis Train images...")
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
img_size = 640
e=1
shapes = []
import time
print("Start Analysis Train image shapes...")
for im_file in img_files:
    im = Image.open(im_file)
    im.verify()  # PIL verify
    shape = im.size  # image size
    #print(shape)
    #shapes.append([shape[0],shape[1]])
    shapes.append(shape)
    #print("\r","",end="\r")
    print('\r        <----',end='\r')
    print('\r',e,end='\r')
    #print("",end='\r')
    #time.sleep(.001)
    e+=1
    if e==stop_num:
        break
#for shape in shapes:
    #print(shape)
    
shapes = np.array(shapes, dtype=np.float64)
shapes = img_size * shapes / shapes.max(1, keepdims=True)
#for shape in shapes:
    #print(shape)
'''
for img_file in img_files:
    print(c,": ",img_file)
    c+=1
'''  
label_files = img2label_paths(img_files)
par = Path(label_files[0]).parent
print("par =",par)
cache_path = r"D:\datasets-smallTLR\coco\train2017.cache"
print("cache_path = ",cache_path)
cache_version = 0.6
try:
    cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
    assert cache['version'] == cache_version  # same version
    #print(cache)
    #assert cache['hash'] == get_hash(self.label_files + self.img_files)  # same hash
except:
    print("A")
    #cache, exists =cache_labels(cache_path, prefix), False  # cache
label_list = []
print("Start Analysis labels...")
for label_file in label_files:
    #print(c," :",label_file)
    c+=1
    print('\r            <---------',end='\r')
    print('\r',c,end='\r')
    if c==stop_num:
        break
    if os.path.isfile(label_file):
        with open(label_file) as f:
            y = f.read().strip().splitlines()
            #print("Alister~~~~~")
            #print("y = ",y)
            l = [x.split() for x in y if len(y)]
            l = np.array(l, dtype=np.float16())
            #nl = len(l)
            #print("l=",l)
            _, i = np.unique(l, axis=0, return_index=True)
            #print("_ = ",_)
            label_list.append(_)
            #print("label_list = ",label_list)
            #if len(i) < nl:  # duplicate row check
                #l = l[i]  # remove duplicates
                #print("l = ",l)
           
#for label in label_list:
    #print("label = ",label)
            
    #else:
        #print("It doesn't exist")
wh0 = np.concatenate([l[:, 3:5]*s for s,l in zip(shapes, label_list)])  # wh
wh0 = np.array(wh0)
#wh0 = [l[:, 3:5] for l in label_list]  # wh
d = 1
#for wh in wh0:
    #print(d," :",wh)
    #d+=1
    
n = 9
print("Filter img size < 2 pixels")
wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels
print("Start using Kmeans to get Anchors")
s = wh.std(0)  # sigmas for whitening
print("Standard Diviation = ",s)
k, dist = kmeans(wh/s, n, iter=30)  # points, mean distance
k *= s
sort_k = k[np.argsort(k.prod(1))]  # sort small to large
for ks in sort_k:
    print(ks[0]," ," ,ks[1])


