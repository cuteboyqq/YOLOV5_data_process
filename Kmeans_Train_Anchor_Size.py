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
import torch
from tqdm import tqdm
import random
stop_num = 83500
path = "/home/ali/datasets-smallTLR/coco/images/train2017"
prefix=''
f = []  # image files
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
c= 1
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of multiprocessing threads
anchor_num = 9
K_means_max_iter = 40

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


def metric(k, wh):  # compute metrics
    r = wh[:, None] / k[None]
    x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
    # x = wh_iou(wh, torch.tensor(k))  # iou metric
    return x, x.max(1)[0]  # x, best_x



def anchor_fitness(k,wh):  # mutation fitness
    thr = 0.25 
    _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
    return (best * (best > thr).float()).mean()  # fitness

def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

def label2img_path(label_path):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return sa.join(label_path.rsplit(sb, 1)).rsplit('.', 1)[0] + '.jpg'

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
#======================================================================================================================
'''
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
'''
for img_file in img_files:
    print(c,": ",img_file)
    c+=1
'''
#========================================================================================================================  
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
#print("Start Analysis labels...")
#pbar = tqdm(range(gen), desc=f'{PREFIX} Analysis labels :')  # progress bar
#for _ in pbar:
PREFIX = colorstr('Generate labels List: ')
for label_file in tqdm(label_files,desc=f'{PREFIX}Start Analysis labels:'):
    #print(c," :",label_file)
    c+=1
    #print('\r            <---------',end='\r')
    #print('\r',c,end='\r')
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
    else:
        img_path = label2img_path(label_file)
        print("remove img from list :", img_path)
        img_files.remove(img_path)
           
#for label in label_list:
    #print("label = ",label)
            
    #else:
        #print("It doesn't exist")
#==============================================================================================
img_size = 640
e=1
shapes = []
import time
#print("Start Analysis Train image shapes...")
PREFIX = colorstr('Generate Image shapes List:')
for im_file in tqdm(img_files,desc =f'{PREFIX}Start Analysis Train image shapes:'):
    im = Image.open(im_file)
    im.verify()  # PIL verify
    shape = im.size  # image size
    #print(shape)
    #shapes.append([shape[0],shape[1]])
    shapes.append(shape)
    #print("\r","",end="\r")
    #print('\r        <----',end='\r')
    #print('\r',e,end='\r')
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
#===========================================================================================================    
print("Start filter labels list : wh  And multiple s , which s = shapes = img_size * shapes / shapes.max(1, keepdims=True) ")    
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

#==============================================
import matplotlib.pyplot as plt
# Plot
PREFIX = colorstr('Generate K-means of "x:num(iteration)--y:Avg.distance" Plot:')
k, d = [None]*K_means_max_iter, [None]*K_means_max_iter
for i in tqdm(range(1, K_means_max_iter+1),desc=f'{PREFIX}plot'):
     k[i-1], d[i-1] = kmeans(wh / s, 9,iter=i)  # points, mean distance
fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
ax = ax.ravel()
ax[0].plot(np.arange(1, K_means_max_iter+1), np.array(d) , marker='.')
fig.savefig('wh.png', dpi=200)   
#==============================================
min_d = 99999
c = 0
min_dis_index = 7777
for dis in d:
    if dis < min_d:
        min_d = dis
        min_dis_index = c        
    c+=1

print("K-means min distance is at index :",min_dis_index)    
#k, dist = kmeans(wh/s, n, iter=30)  # points, mean distance
k = k[min_dis_index]
k *= s
sort_k = k[np.argsort(k.prod(1))]  # sort small to large
print("After K-means...Anchor Values")
for ks in sort_k:
    print(ks[0]," ," ,ks[1])



    
    
print("Start Generic Algorithm...")
# Evolve
npr = np.random
gen=1000
PREFIX = colorstr('AutoAnchor: ')
wh = torch.tensor(wh, dtype=torch.float32)  # filtered
f, sh, mp, s = anchor_fitness(k,wh), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
pbar = tqdm(range(gen), desc=f'{PREFIX}Evolving anchors with Genetic Algorithm:')  # progress bar
for _ in pbar:
    v = np.ones(sh)
    while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
        v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
    kg = (k.copy() * v).clip(min=2.0)
    fg = anchor_fitness(kg,wh)
    if fg > f:
        f, k = fg, kg.copy()
        pbar.desc = f'{PREFIX}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
print("")
print("After Generic Algorithm...Anchor Values")
sort_k = k[np.argsort(k.prod(1))]  # sort small to large
for ks in sort_k:
    print(ks[0]," ," ,ks[1])

'''
import matplotlib.pyplot as plt
# Plot
PREFIX = colorstr('Generate Plot:')
k, d = [None]*10, [None]*10
for i in tqdm(range(1, 11),desc=f'{PREFIX}plot'):
     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
ax = ax.ravel()
ax[0].plot(np.arange(1, 11), np.array(d) , marker='.')
fig.savefig('wh.png', dpi=200)    
'''

