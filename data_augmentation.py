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
import random
import cv2
import tqdm


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


def load_image(img_path,img_size,resize=False):
    img = cv2.imread(img_path) #BGR
    assert img is not None, 'Image Not Found' + img_path
    h0,w0 = img.shape[:2]
    r = img_size / max(h0,w0) #resize image to image_size
    if resize==True:
        if r!=1: # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r<1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0*r),int(h0*r)), interpolation=interp)
    return img, (h0,w0), img.shape[:2] #img, hw_original, hw_resized        

def img2label_paths(img_paths):
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep
    return [ sb.join(x.rsplit(sa,1)).rsplit('.')[0] + '.txt' for x in img_paths]

def img2label_path(img_path,folder_name='new_labels',f_name='_2'):
    sa, sb = os.sep + 'images' + os.sep, os.sep + folder_name + os.sep
    return sb.join(img_path.rsplit(sa,1)).rsplit('.')[0] + f_name + '.txt'


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5, do_he=False, num=6):
    r = np.random.uniform(-1,1,3) * [hgain, sgain, vgain] + 1 #random gains
    
    img = np.array(img)
    
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype # uint8
    
    x = np.arange(0, 256, dtype=np.int16)
    lut_h = ((x * r[0]) % 180).astype(dtype)
    lut_s = np.clip(x * r[1], 0, 255).astype(dtype)
    #lut_v = np.clip(x * r[2], 0, 255).astype(dtype)
    for i in range(num):
        ra = (i+1) *  (float)(1/num)
        lut_v = np.clip(x * ra, 0, 255).astype(dtype)
    
        img_hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v))).astype(dtype)
    #img_hsv = cv2.merge((h, s, cv2.LUT(v, lut_v))).astype(dtype)
       
        if i==1:
            result_img_2 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR) #no return needed
        elif i==2:
            result_img_3 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR) #no return needed
        elif i==3:
            result_img_4 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR) #no return needed
        elif i==4:
            result_img_5 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR) #no return needed
        elif i==5:
            result_img_6 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR) #no return needed
        elif i==6:
            result_img_7 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR) #no return needed
        elif i==7:
            result_img_8 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR) #no return needed
        elif i==8:
            result_img_9 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR) #no return needed
    
    #result_he_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR) #no return needed
    #Histogram equalization
        if do_he==True:
            if random.random() < 0.2:
                for i in range(3):
                    result_img2[:,:,i] = cv2.equalizeHist(result_img2[:,:,i])
    if num==3:
        return result_img_2
    elif num==4:
        return result_img_2,result_img_3
    elif num==5:
        return result_img_2,result_img_3,result_img_4
    elif num==6:
        return result_img_2,result_img_3,result_img_4,result_img_5
    elif num==7:
        return result_img_2,result_img_3,result_img_4,result_img_5,result_img_6
    elif num==8:
        return result_img_2,result_img_3,result_img_4,result_img_5,result_img_6,result_img_7
    elif num==9:
        return result_img_2,result_img_3,result_img_4,result_img_5,result_img_6,result_img_7,result_img_8
    elif num==10:
        return result_img_2,result_img_3,result_img_4,result_img_5,result_img_6,result_img_7,result_img_8,result_img_9
   
    '''
    if do_he==True:
        return result_img,result_he_img
    else:
        return result_img
    '''
def Analysis_Path(path):
    path_dir = os.path.dirname(path)
    path_dir_dir = os.path.dirname(path_dir)
    f = path.split(os.sep)[-1]
    f_name = f.split(".")[0]
    return path_dir,f,f_name,path_dir_dir
def Data_Augmentation(path,
                      img_size=640,
                      max_num=200000,
                      hsv_aug=True,
                      data_type=0,
                      num_v = 10):
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
    c = 1
    aug_img_files = []
    
    
    pbar = tqdm.tqdm(img_files)
    for img_f in pbar:
        #print(img_f)
        img, _, (h,w) = load_image(img_f,img_size)
        #print(_," (",h,",",w,")")
        
        PREFIX = colorstr('HSV-augmentation:')
        pbar.desc = f'{PREFIX}'
        
        if hsv_aug == True:
            if num_v==10:
                result_img_2,result_img_3,result_img_4,result_img_5,result_img_6,result_img_7,result_img_8,result_img_9 = augment_hsv(img,0,0,0.9,False,num_v)
            elif num_v==9:
                result_img_2,result_img_3,result_img_4,result_img_5,result_img_6,result_img_7,result_img_8 = augment_hsv(img,0,0,0.9,False,num_v)
            elif num_v==8:
                result_img_2,result_img_3,result_img_4,result_img_5,result_img_6,result_img_7 = augment_hsv(img,0,0,0.9,False,num_v)
            elif num_v==7:
                result_img_2,result_img_3,result_img_4,result_img_5,result_img_6 = augment_hsv(img,0,0,0.9,False,num_v)
            elif num_v==6:
                result_img_2,result_img_3,result_img_4,result_img_5 = augment_hsv(img,0,0,0.9,False,num_v)
            elif num_v==5:
                result_img_2,result_img_3,result_img_4 = augment_hsv(img,0,0,0.9,False,num_v)
            elif num_v==4:
                result_img_2,result_img_3 = augment_hsv(img,0,0,0.9,False,num_v)
            elif num_v==3:
                result_img_2 = augment_hsv(img,0,0,0.9,False,num_v)
            #result_img_2 = augment_hsv(img,0,0,0.4)
         
            name_ori = str(c) + '_ori.jpg'
            #name_he = str(c) + '_he.jpg'
            
            label_f = img2label_path(img_f,'labels','')
            
            img_dir,f,f_name,path_dir_dir = Analysis_Path(img_f)
            '''========================================================================================'''
            for i in range(num_v):
                if data_type == 0: #train/val
                    if i%2==0 and i>0:
                        name = f_name + '_' + str(i) + '.jpg'
                        folder_name = 'train'
                        path_dir = os.path.join(path_dir_dir,'aug','images',folder_name)
                        if not os.path.exists(path_dir):
                            os.makedirs(path_dir)
                        path = os.path.join(path_dir,name)
                        #img = 'result_img_'+str(i)
                        if c < max_num:
                            if i==2:
                                cv2.imwrite(path,result_img_2)
                            elif i==4:
                                cv2.imwrite(path,result_img_4)
                            elif i==6:
                                cv2.imwrite(path,result_img_6)
                            elif i==8:
                                cv2.imwrite(path,result_img_8)
                        add = ''
                        label_f_2 = img2label_path(path,'labels',add)        
                        l2_dir,l2f,l2f_name,l2path_dir_dir = Analysis_Path(label_f_2)
                        if not os.path.exists(l2_dir):
                            os.makedirs(l2_dir)
                        if os.path.exists(label_f):
                            shutil.copy(label_f,label_f_2)       
                        
                    elif i%2==1:
                        name = f_name + '_' + str(i) + '.jpg'
                        folder_name = 'val'
                        path_dir = os.path.join(path_dir_dir,'aug','images',folder_name)
                        if not os.path.exists(path_dir):
                            os.makedirs(path_dir)
                        path = os.path.join(path_dir,name)
                        
                        if c < max_num:
                            if i==3:
                                cv2.imwrite(path,result_img_3)
                            elif i==5:
                                cv2.imwrite(path,result_img_5)
                            elif i==7:
                                cv2.imwrite(path,result_img_7)
                            elif i==9:
                                cv2.imwrite(path,result_img_9)
                        if i>1:        
                            add = ''
                            label_f_2 = img2label_path(path,'labels',add)        
                            l2_dir,l2f,l2f_name,l2path_dir_dir = Analysis_Path(label_f_2)
                            if not os.path.exists(l2_dir):
                                os.makedirs(l2_dir)
                            if os.path.exists(label_f):
                                shutil.copy(label_f,label_f_2) 
                        #img = 'result_img_'+str(i)
                        #if c < max_num:
                            #cv2.imwrite(path,img)
                elif data_type == 1 or data_type == 2: #train or val only
                    name = f_name + '_' + str(i) + '.jpg'
                    if data_type == 1:
                        folder_name = 'train'
                    else:
                        folder_name = 'val'
                    path_dir = os.path.join(path_dir_dir,'aug','images',folder_name)
                    if not os.path.exists(path_dir):
                        os.makedirs(path_dir)
                    path = os.path.join(path_dir,name)
                    #img = 'result_img_'+str(i)
                    if c < max_num:
                        if i==2:
                            cv2.imwrite(path,result_img_2)
                        elif i==4:
                            cv2.imwrite(path,result_img_4)
                        elif i==6:
                            cv2.imwrite(path,result_img_6)
                        elif i==8:
                            cv2.imwrite(path,result_img_8)
                        elif i==9:
                            cv2.imwrite(path,result_img_9)
                        elif i==3:
                            cv2.imwrite(path,result_img_3)
                        elif i==5:
                            cv2.imwrite(path,result_img_5)
                        elif i==7:
                            cv2.imwrite(path,result_img_7)
                    if i>1: 
                        add = ''
                        label_f_2 = img2label_path(path,'labels',add)        
                        l2_dir,l2f,l2f_name,l2path_dir_dir = Analysis_Path(label_f_2)
                        if not os.path.exists(l2_dir):
                            os.makedirs(l2_dir)
                        if os.path.exists(label_f):
                            shutil.copy(label_f,label_f_2)       
            '''======================================================================================'''
            '''
            name2 =  f_name +'_2.jpg'
            path2_dir = os.path.join(path_dir_dir,'val_aug2')
            if not os.path.exists(path2_dir):
                os.makedirs(path2_dir)
            path2 = os.path.join(path2_dir,name2)
            
            label_f_2 = img2label_path(img_f,'labels_aug2','_2')
            l2_dir,l2f,l2f_name,l2path_dir_dir = Analysis_Path(label_f_2)
            if not os.path.exists(l2_dir):
                os.makedirs(l2_dir)
            #shutil.copy(label_f,label_f_2)
          
                
            name3 =  f_name +'_3.jpg'
            path3_dir = os.path.join(path_dir_dir,'train_aug3')
            if not os.path.exists(path3_dir):
                os.makedirs(path3_dir)
            path3 = os.path.join(path3_dir,name3)
            
            label_f_3 = img2label_path(img_f,'labels_aug3','_3')
            l3_dir,l3f,l3f_name,l3path_dir_dir = Analysis_Path(label_f_3)
            if not os.path.exists(l3_dir):
                os.makedirs(l3_dir)
            #shutil.copy(label_f,label_f_3)
           
            name4 =  f_name +'_4.jpg'
            path4_dir = os.path.join(path_dir_dir,'val_aug4')
            if not os.path.exists(path4_dir):
                os.makedirs(path4_dir)
            path4 = os.path.join(path4_dir,name4)
            
            label_f_4 = img2label_path(img_f,'labels_aug4','_4')
            l4_dir,l4f,l4f_name,l4path_dir_dir = Analysis_Path(label_f_4)
            if not os.path.exists(l4_dir):
                os.makedirs(l4_dir)
            #shutil.copy(label_f,label_f_4)
          
            name5 =  f_name +'_5.jpg'
            path5_dir = os.path.join(path_dir_dir,'train_aug5')
            if not os.path.exists(path5_dir):
                os.makedirs(path5_dir)
            path5 = os.path.join(path5_dir,name5)
            
            label_f_5 = img2label_path(img_f,'labels_aug5','_5')
            l5_dir,l5f,l5f_name,l5path_dir_dir = Analysis_Path(label_f_5)
            if not os.path.exists(l5_dir):
                os.makedirs(l5_dir)
            #shutil.copy(label_f,label_f_5)
          
            name6 =  f_name +'_6.jpg'
            path6_dir = os.path.join(path_dir_dir,'val_aug6')
            if not os.path.exists(path6_dir):
                os.makedirs(path6_dir)
            path6 = os.path.join(path6_dir,name6)
            
            label_f_6 = img2label_path(img_f,'labels_aug6','_6')
            l6_dir,l6f,l6f_name,l6path_dir_dir = Analysis_Path(label_f_6)
            if not os.path.exists(l6_dir):
                os.makedirs(l6_dir)
            #shutil.copy(label_f,label_f_5)
            '''
            
            #print(img_dir)
            #print(f)
            #print(f_name)
            '''
            if c < max_num:
                #cv2.imwrite(name,result_img_1)
                cv2.imwrite(path2,result_img_2)
                cv2.imwrite(path3,result_img_3)
                cv2.imwrite(path4,result_img_4)
                cv2.imwrite(path5,result_img_5)
                #cv2.imwrite(name2,result_img_2)
                cv2.imwrite(name_ori,img)
                
                
                shutil.copy(label_f,label_f_2)
                shutil.copy(label_f,label_f_3)
                shutil.copy(label_f,label_f_4)
                shutil.copy(label_f,label_f_5)
                #cv2.imwrite(name_he,result_he_img)
            '''
            c+=1
    label_files = img2label_paths(img_files)
    
    #for label_f in label_files:
        #print(label_f)
    
        



if __name__=="__main__":
    path =  "/home/ali/datasets/BSTLD_data/train/images"
    img_size = 320
    Data_Augmentation(path,
                      img_size,
                      200000,
                      True,
                      1,
                      7)
    