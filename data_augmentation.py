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



def augment_flip(img,flip_type=1):
    
    img = np.array(img)
    if flip_type==0: #flip up/down and left/right
        flip_lr_img = cv2.flip(img,1)
        flip_ud_img = cv2.flip(img,0)
        return flip_lr_img,flip_ud_img
    elif flip_type==1: #flip lr
        flip_lr_img = cv2.flip(img,1)
        return flip_lr_img
    elif flip_type==2: #flip up/down
        flip_ud_img = cv2.flip(img,0)
        return flip_ud_img
    
    



def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5, do_he=False, num=6):
    r = np.random.uniform(-1,1,3) * [hgain, sgain, vgain] + 1 #random gains
    
    img = np.array(img)
    
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype # uint8
    
    x = np.arange(0, 256, dtype=np.int16)
    lut_h = ((x * r[0]) % 180).astype(dtype)
    lut_s = np.clip(x * r[1], 0, 255).astype(dtype)
    #lut_v = np.clip(x * r[2], 0, 255).astype(dtype)
    hsv_images = []
    hsv_he_images = []
    for i in range(num):
        ra = (i+1) *  (float)(1/num)
        lut_v = np.clip(x * ra, 0, 255).astype(dtype)
    
        img_hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v))).astype(dtype)
    #img_hsv = cv2.merge((h, s, cv2.LUT(v, lut_v))).astype(dtype)
        result_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        hsv_images.append(result_img)
        result_he_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR) #no return needed
    #Histogram equalization
        if do_he==True:
            if random.random() < 0.2:
                for i in range(3):
                    result_he_img[:,:,i] = cv2.equalizeHist(result_img[:,:,i])
            
            else:
                result_he_img = result_img
            
            hsv_he_images.append(result_he_img)
    
    return hsv_images,hsv_he_images
            
    
   



def Analysis_Path(path):
    path_dir = os.path.dirname(path)
    path_dir_dir = os.path.dirname(path_dir)
    f = path.split(os.sep)[-1]
    f_name = f.split(".")[0]
    return path_dir,f,f_name,path_dir_dir

def save_flip_image(img_f,flip_lr_img,data_type=1,flip_type=1,flip_folder_name='aug_flip'):
    img_dir,f,f_name,path_dir_dir = Analysis_Path(img_f)
    if flip_type == 1:
        name = f_name + '_flr.jpg'
    elif flip_type == 0:
        name = f_name + '_fudlr.jpg'
    elif flip_type == 2:
        name = f_name + '_fud.jpg'
    
    if data_type==1 or data_type==0 :
        path_dir = os.path.join(path_dir_dir,flip_folder_name,'images','train')
    elif data_type==2:
        path_dir = os.path.join(path_dir_dir,flip_folder_name,'images','val')
    
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
        
    save_path = os.path.join(path_dir,name)
    
    cv2.imwrite(save_path,flip_lr_img)


def save_flip_label(img_f,ori_class,flip_class,data_type=1,flip_type=1,flip_folder_name='aug_flip'):
    
    label_f = img2label_path(img_f,'labels','')
    img_dir,f,f_name,path_dir_dir = Analysis_Path(img_f)
    
    
    if data_type==0 or data_type==1: #train/val  or train only
        path_dir = os.path.join(path_dir_dir,flip_folder_name,'labels','train')
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        if flip_type==1:
            label_flip_f = f_name + '_flr.txt'
        elif flip_type==2:
            label_flip_f = f_name + '_fud.txt'
        elif flip_type==0:
            label_flip_f = f_name + '_flrud.txt'
        
        flip_label_path = os.path.join(path_dir,label_flip_f)
        
        
    f_label_flip = open(flip_label_path,'a+')
    if os.path.exists(label_f):
        f = open(label_f,'r')
        lines = f.readlines()
        for l in lines: #l=[c,x,y,w,h]
            #print(l)
            l_list = l.split(" ")
            #print(l_list)
            '''x_flip = 1 - x'''
            float_l_list_1 = 1.0 - float(l_list[1])
            int_l_list_1 = int((1.0 - float(l_list[1]))*1000000)
            final_l_list_1 = float(int_l_list_1/1000000)
            l_list[1] = str( final_l_list_1 ) #modify x
            
            ''' some labels(c) need to change, ex: (c)left label --> right label(c_flip) '''
            c = int(l_list[0])
            c_flip = flip_class[c]
            l_list[0] = str(c_flip)#modify c
            #========================================================================
            l_final = " ".join(l_list) #list to string
            f_label_flip.write(l_final) #write to .txt file
            '''
            try:
                l[1] = str((1.0 - float(l[1])))
                f_label_flip.write(l)
            except ValueError:
                print("error")
            '''
    f_label_flip.close()
    
    return True



def save_hsv_image_and_label(img_f,num_v,data_type,c,max_num,hsv_images,min_th,hsv_folder_name='aug_flip'):
    
    save_complete = False
    img_dir,f,f_name,path_dir_dir = Analysis_Path(img_f)
    label_f = img2label_path(img_f,'labels','')
    '''========================================================================================'''
    for i in range(num_v):
        if data_type == 0: #train/val
            if i%2==0 and i>0: # odd for train
                name = f_name + '_' + str(i) + '.jpg'
                folder_name = 'train'
                path_dir = os.path.join(path_dir_dir,hsv_folder_name,'images',folder_name)
                if not os.path.exists(path_dir):
                    os.makedirs(path_dir)
                path = os.path.join(path_dir,name)
                #img = 'result_img_'+str(i)
                if c < max_num:
                    if i>=min_th and i<= (num_v-2):
                        cv2.imwrite(path,hsv_images[i])
                 
                if i>=min_th and i<=(num_v - 2) : 
                    add = ''
                    label_f_2 = img2label_path(path,'labels',add)        
                    l2_dir,l2f,l2f_name,l2path_dir_dir = Analysis_Path(label_f_2)
                    if not os.path.exists(l2_dir):
                        os.makedirs(l2_dir)
                    if os.path.exists(label_f):
                        shutil.copy(label_f,label_f_2)       
                
            elif i%2==1: #even for val
                name = f_name + '_' + str(i) + '.jpg'
                folder_name = 'val'
                path_dir = os.path.join(path_dir_dir,hsv_folder_name,'images',folder_name)
                if not os.path.exists(path_dir):
                    os.makedirs(path_dir)
                path = os.path.join(path_dir,name)
                
                if c < max_num:
                    if i>=min_th and i<=(num_v-1):
                        cv2.imwrite(path,hsv_images[i])
                 
                if i>=min_th and i<=(num_v-1):        
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
            path_dir = os.path.join(path_dir_dir,hsv_folder_name,'images',folder_name)
            if not os.path.exists(path_dir):
                os.makedirs(path_dir)
            path = os.path.join(path_dir,name)
            #img = 'result_img_'+str(i)
            if c < max_num:
                if i>=min_th and i<=(num_v-1):
                    cv2.imwrite(path,hsv_images[i])
             
            if i>=min_th and i<=(num_v - 1): 
                add = ''
                label_f_2 = img2label_path(path,'labels',add)        
                l2_dir,l2f,l2f_name,l2path_dir_dir = Analysis_Path(label_f_2)
                if not os.path.exists(l2_dir):
                    os.makedirs(l2_dir)
                if os.path.exists(label_f):
                    shutil.copy(label_f,label_f_2)    
                
    '''==============================End hsv augmentation========================================================'''
    save_complete=True
    return save_complete



def augment_blur(img,blur_type,blur_size,view_blurimg):
    
    #img = np.array(img)
    im = cv2.imread(img)    
    img_mean = cv2.blur(im,(blur_size,blur_size))
    img_Guassian = cv2.GaussianBlur(im,(blur_size,blur_size),0)
    img_median = cv2.medianBlur(im,blur_size)
    img_bilater = cv2.bilateralFilter(im,9,75,75)
        
    titles = ['srcImg',   'mean'  ,  'Gaussian',    'median',   'bilateral']
    imgs =   [  im    ,img_mean   , img_Guassian,  img_median,   img_bilater]
    
    if view_blurimg:
        import matplotlib.pyplot as plt
        for i in range(5):
            plt.subplot(2,3,i+1)
            plt.imshow(imgs[i])
            plt.title(titles[i])
        plt.show()
    return imgs,titles

def save_blur_img_and_labels(img,img_f,blur_type,blur_size,data_type,blur_name):
    
    saved = False
    img_dir,f,f_name,path_dir_dir = Analysis_Path(img_f)
    label_f = img2label_path(img_f,'labels','')
    la_dir,la_f,la_f_name,la_path_dir_dir = Analysis_Path(label_f)
    
    if blur_type>=1 and blur_type<=4:
        folder_name = 'blur_' + blur_name[blur_type] + '_' + str(blur_size)
        name = f_name + '_' + blur_name[blur_type] +'blur_' + str(blur_size) + '.jpg'
        l_name = f_name + '_' + blur_name[blur_type] + 'blur_' + str(blur_size) + '.txt'
        save_img = img[blur_type]
    else:
        return saved
    
    '''
    if blur_type==1:
        folder_name = 'blur_mean'
        name = f_name + '_meanblur_' + str(blur_size) + '.jpg'
        
        l_name = f_name + '_meanblur_' + str(blur_size) + '.txt'
        save_img = img[1]
    elif blur_type==2:
        folder_name = 'blur_Gaussian'
        name = f_name + '_Gaussianblur_' + str(blur_size) + '.jpg' 
        l_name = f_name + '_Gaussianblur_' + str(blur_size) + '.txt' 
        save_img = img[2]
    elif blur_type==3:
        folder_name = 'blur_median'
        name = f_name + '_medianblur_' + str(blur_size) + '.jpg' 
        l_name = f_name + '_medianblur_' + str(blur_size) + '.txt' 
        save_img = img[3]
    elif blur_type==4:
        folder_name = 'blur_bilateral'
        name = f_name + '_bilateralblur_' + str(blur_size) + '.jpg' 
        l_name = f_name + '_bilateralblur_' + str(blur_size) + '.txt' 
        save_img = img[4]
    '''   
    if data_type==1 or data_type==0:
        path_dir = os.path.join(path_dir_dir,folder_name,'images','train')
        label_path_dir = os.path.join(path_dir_dir,folder_name,'labels','train')
    elif data_type==2:
        path_dir = os.path.join(path_dir_dir,folder_name,'images','val')
        label_path_dir = os.path.join(path_dir_dir,folder_name,'labels','val')
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    
    if not os.path.exists(label_path_dir):
        os.makedirs(label_path_dir)
        
    
    path = os.path.join(path_dir,name)
    cv2.imwrite(path,save_img)
    
    label_path = os.path.join(label_path_dir,l_name)
    
    if os.path.exists(label_f):
        if not os.path.exists(label_path):
            shutil.copy(label_f,label_path)
    
    saved = True
    
    return saved
    

def Data_Augmentation(path,
                      ori_class,
                      flip_class,
                      img_size=640,
                      max_num=200000,
                      hsv_aug=True,
                      hsv_folder_name = "aug_hsv",
                      flip_aug=True,
                      flip_type = 1,
                      flip_folder_name = "aug_flip",
                    
                      data_type=1,
                      num_v = 10,
                      min_th = 10,
                      blur_aug=True,
                      blur_type=1,
                      blur_size=5,
                      view_blurimg=True):
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
        bar_str = ''
        '''==============================Start hsv augmentation====================================================='''
        if hsv_aug == True:
            bar_str += ' Hsv '
            hsv_images,hsv_he_images = augment_hsv(img,0,0,0.9,True,num_v)
            #print(hsv_images)
            #print(hsv_he_images)
            name_ori = str(c) + '_ori.jpg'
            #name_he = str(c) + '_he.jpg'
            save_complete = save_hsv_image_and_label(img_f,num_v,data_type,c,max_num,hsv_images,min_th,hsv_folder_name)
        '''================================Start flip augmentation==============================================='''
        if flip_aug == True:
            bar_str += ' Flip '
            flip_lr_img = augment_flip(img,1)
            save_flip_image(img_f,flip_lr_img,data_type,flip_type,flip_folder_name)
            labell = save_flip_label(img_f,ori_class, flip_class,data_type,flip_type,flip_folder_name)
                                 
        '''==============================Start Blur augmentation===================================================='''
        if blur_aug == True:
            bar_str += ' Blur '
            img_blur,blur_name = augment_blur(img_f,blur_type,blur_size,view_blurimg) #return img list
            save_blur = save_blur_img_and_labels(img_blur,img_f,blur_type,blur_size,data_type,blur_name)
        '''========================================================================================================='''
        bar_str += ' Augmentation:'
        PREFIX = colorstr(bar_str)
        pbar.desc = f'{PREFIX}'
        c+=1
        if c > max_num:
            break
    label_files = img2label_paths(img_files)
    
    #for label_f in label_files:
        #print(label_f)
    
def get_args():
    import argparse
    
    parser = argparse.ArgumentParser()
    #'/home/ali/datasets/train_video/NewYork_train/train/images'
    parser.add_argument('-imgdir','--img-dir',help='image dir',default='/home/ali/datasets/train_video/NewYork_train/train/images')
    parser.add_argument('-imgsize','--img-size',type=int,help='image size',default=320)
    parser.add_argument('-maxnum','--max-num',type=int,help='max number of analysis images',default=200000)
    parser.add_argument('-hsvaug','--hsv-aug',action='store_true',help='do hsv augment')
    parser.add_argument('-hsvname','--hsv-name',help='hsv aug folder name',default='aug_hsv')
    parser.add_argument('-flipaug','--flip-aug',action='store_true',help='do flip augment')
    parser.add_argument('-flipname','--flip-name',help='flip aug folder name',default='aug_flip')
    parser.add_argument('-fliptype','--flip-type',type=int,default=1,help='0:flip lr/ud, 1:flip lr, 2:flip ud')
    parser.add_argument('-datatype','--data-type',type=int,default=0,help='0:train/val, 1:train, 2:val')
    parser.add_argument('-numhsvclass','--num-hsvclass',type=int,default=12,help='num of hsv class')
    parser.add_argument('-minhsvclass','--min-hsvclass',type=int,default=5,help='min hsv class')
    parser.add_argument('-blurtype','--blur-type',type=int,default=2,help='blur type : 0:original; 1:mean; 2:Gaussian; 3:median; 4:bilateral')
    parser.add_argument('-blursize','--blur-size',type=int,default=9,help='filter size of blur')
    parser.add_argument('-bluraug','--blur-aug',action='store_true',help='do blur augment')
    parser.add_argument('-viewblur','--view-blur',action='store_true',help='view blur images')
    
    return parser.parse_args()    

if __name__=="__main__":
    path =  "/home/ali/datasets/LISA_data/train/13/direction_TL/images"
    img_size = 320
    ori_class= [x for x in range(17)]
    flip_class = [0,1,2,3,4,5,6,7,8,9,14,11,12,15,10,13,16]
    
    
    args = get_args()
    
    path = args.img_dir
    img_size = args.img_size
    max_num = args.max_num
    do_hsv_aug = args.hsv_aug
    hsv_folder_name = args.hsv_name
    do_flip_aug = args.flip_aug
    flip_type = args.flip_type
    flip_folder_name = args.flip_name
    data_type = args.data_type
    num_hsv_class = args.num_hsvclass
    min_hsvclass = args.min_hsvclass
    blur_type = args.blur_type
    blur_size = args.blur_size
    do_blur_aug = args.blur_aug
    view_blur_imgs = args.view_blur
    
    blur_name = ['srcImg',   'mean'  ,  'Gaussian',    'median',   'bilateral']
    flip_name = ['left/right&up/down','left/right','up/down']
    data_name = ['train/val','train','val']
    print("path = ",path)
    print("img_size = ",img_size)
    print("max_num = ",max_num)
    print("data_type =",data_name[data_type])
    
    print("==========================================")
    print("HSV-Augment:")
    print("do_hsv_aug =",do_hsv_aug)
    print("num_hsv_class =",num_hsv_class)
    print("hsv_folder_name = ",hsv_folder_name)
    print("min_hsvclass = ",min_hsvclass)
    
    print("==========================================")
    print("Flip Augment:")
    print("do_flip_aug =",do_flip_aug)
    print("flip_type =",flip_name[flip_type])
    print("flip_folder_name =",flip_folder_name)
    
    print("==========================================")
    print("Blur Augment:")
    print("do_blur_aug =",do_blur_aug)
    print("blur_type = ",blur_name[blur_type])
    print("blur_size = ",blur_size)
    print("view_blur_imgs = ",view_blur_imgs)
    Data_Augmentation(path,
                      ori_class, #original class label
                      flip_class, #flip class label
                      img_size, #final size, (not used)
                      max_num, #max num
                      True, #do_hsv_aug do hsv augment
                      hsv_folder_name, #hsv aug save folder name
                      False, #do_flip_aug do flip augment
                      flip_type, #flip type --> 0:lr+ud, 1:lr, 2:ud
                      flip_folder_name, # flip aug save folder name
                      data_type, # data type --> 0:train/val 1:train 2:val
                      num_hsv_class,# number of hsv aug labels , remove front and tail labels
                      min_hsvclass,# min_th
                      False,# do blur_aug
                      blur_type,#blur_type : 0:ori, 1:mean, 2:Gaussian, 3:median, 4:bilateral,5:all
                      blur_size, #blur size
                      False) #view_blurimg
    