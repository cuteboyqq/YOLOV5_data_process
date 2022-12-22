# -*- coding: utf-8 -*-
"""
Created on Sun May  8 12:59:13 2022

@author: User
"""
import os
import shutil
import cv2
import glob
import tqdm
import numpy as np
import random

def Analysis_Img_Path(img_path):
    img_dir = os.path.dirname(img_path)
    img_dir_name = os.path.basename(img_dir)
    img = img_path.split(os.sep)[-1]
    img_name = img.split(".jpg")[0]
    
    print(img_name)
    return img_dir_name,img,img_name

    

def augment_hsv(img_path, save_img_dir, hgain=0.5, sgain=0.5, vgain=0.5, do_he=False, num=10):
    print("aug_hsv not implemented")
    r = np.random.uniform(-1,1,3) * [hgain, sgain, vgain] + 1 #random gains
    
    img = cv2.imread(img_path)
    img = np.array(img)
    
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype # uint8
    
    x = np.arange(0, 256, dtype=np.int16)
    lut_h = ((x * r[0]) % 180).astype(dtype)
    lut_s = np.clip(x * r[1], 0, 255).astype(dtype)
    #lut_v = np.clip(x * r[2], 0, 255).astype(dtype)
    hsv_images = []
    hsv_he_images = []
    '''====================================================================='''
    ori_lable_name, img_file, img_name = Analysis_Img_Path(img_path)
    new_folder_name = ori_lable_name
    save_img_dir = save_img_dir + '_' +  'hsv' 
    
    save_dir = os.path.join(save_img_dir,new_folder_name)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    '''======================================================================'''
    num = int(num)
    for i in range(num*2):
        ra = (i) *  (float)(1/num)
        lut_v = np.clip(x * ra, 0, 255).astype(dtype)
    
        img_hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v))).astype(dtype)
    #img_hsv = cv2.merge((h, s, cv2.LUT(v, lut_v))).astype(dtype)
        result_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        hsv_images.append(result_img)
        result_he_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR) #no return needed
        
        
        '''===================================save image====================================='''
        
        save_img_file = img_name + '_' + str(i) + '.jpg'
        save_img_file_path = os.path.join(save_dir,save_img_file)
        
        if (i) > (num/5) and i!=(num): 
            cv2.imwrite(save_img_file_path,result_img)
        '''===================================save image====================================='''
        
    #Histogram equalization
        if do_he==True:
            if random.random() < 0.2:
                for i in range(3):
                    result_he_img[:,:,i] = cv2.equalizeHist(result_img[:,:,i])
            
            else:
                result_he_img = result_img
            
            hsv_he_images.append(result_he_img)
    
    return hsv_images,hsv_he_images


def aug_blur(img_path,save_img_dir,blur_type,blur_size):
    print("aug_blur not implemented")
    
    im = cv2.imread(img_path)
    img_mean  = cv2.blur(im,(blur_size,blur_size))
    img_Gaussian = cv2.GaussianBlur(im,(blur_size,blur_size),0)
    img_median = cv2.medianBlur(im,blur_size)
    img_bilater = cv2.bilateralFilter(im,9,75,75)
    
    titles = ['srcImg', 'mean', 'Gaussian', 'median', 'bilateral']
    imgs =   [im, img_mean, img_Gaussian, img_median , img_bilater]
    
    ori_lable_name, img, img_name = Analysis_Img_Path(img_path)
  
    new_folder_name = ori_lable_name
    save_img_dir = save_img_dir + '_' +  titles[blur_type] + '_' + str(blur_size)
    
    save_dir = os.path.join(save_img_dir,new_folder_name)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_img_file = img_name + '_' + titles[blur_type] + '_' + str(blur_size) + '.jpg'
    save_img_file_path = os.path.join(save_dir,save_img_file)
    
    cv2.imwrite(save_img_file_path,imgs[blur_type])
    
    return imgs,titles
    
def auf_flip(img_path,save_img_dir,flip_type):
    print(" auf_flip not implemented")
    img = cv2.imread(img_path) #BGR
    img = np.array(img)
    
    img_flip_lrud = cv2.flip(img,-1)
    img_flip_lr = cv2.flip(img,1)
    img_flip_ud = cv2.flip(img,0)
    
    titles = ['flip_lrud', 'flip_lr', 'flip_ud']
    imgs =   [img_flip_lrud, img_flip_lr, img_flip_ud]
    
    ori_lable_name, img, img_name = Analysis_Img_Path(img_path)
    
    new_folder_name = ori_lable_name
    
    save_img_dir = save_img_dir + '_' +  titles[flip_type] 
    
    save_dir = os.path.join(save_img_dir,new_folder_name)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_img_file = img_name + '_' + titles[flip_type] + '.jpg'
    save_img_file_path = os.path.join(save_dir,save_img_file)
    
    
    cv2.imwrite(save_img_file_path,imgs[flip_type])

def pure_img_augmentation(do_blur,blur_type,blur_size,
                          do_flip,flip_type,
                          do_hsv,
                          numv,
                          img_dir,
                          save_img_dir):
    #print("not implemented")
    hsv_images,hsv_he_images = [],[]
    img_path_list = glob.iglob(os.path.join(img_dir,'**/*.jpg'))
    for img_path in img_path_list:
        print(img_path)
        if do_blur:
            imgs, titles = aug_blur(img_path,save_img_dir,blur_type,blur_size)
            
            #print("not implemented")
            
        if do_flip:
            auf_flip(img_path,save_img_dir,flip_type)
            print("not implemented")
        
        if do_hsv:
            hsv_images,hsv_he_images = augment_hsv(img_path, save_img_dir, hgain=0.0, sgain=0.0, vgain=0.9, do_he=False, num=numv)
        
    

def get_args():
    import argparse
    
    parser = argparse.ArgumentParser()
    
    '''============================input img/output img parameters setting================================='''
    parser.add_argument('-imgdir','--img-dir',help='image dir',default='/home/ali/datasets/factory_data/2022-12-21-4cls-cropimg/crops_line')
    parser.add_argument('-savedir','--save-dir',help='save aug-img dir',default='/home/ali/datasets/factory_data/2022-12-21-4cls-cropimg/crops_line_aug')
    
    '''===================blur parameter settings=========================================================='''
    parser.add_argument('-blur','--blur',help='enable blur augment',action='store_true')
    parser.add_argument('-blurtype','--blur-type',help='blur type : 0:original; 1:mean; 2:Gaussian; 3:median; 4:bilateral',default=2)
    parser.add_argument('-blursize','--blur-size',help='blur size',default=11)
    '''===================flip parameter settings=========================================================='''
    parser.add_argument('-flip','--flip',help='enable flip augment',action='store_true')
    parser.add_argument('-fliptype','--flip-type',help='flip type: 0:lrud, 1:lr, 2:ud' ,default=1)
    '''===================hsv parameter settings=========================================================='''
    parser.add_argument('-hsv','--hsv',help='enable hsv augment',action='store_true')
    parser.add_argument('-numv','--numv',help='num of v' ,default=3)
    
    return parser.parse_args()

if __name__=="__main__":
    
    
    args = get_args()
    print("===========IO settings================")
    img_dir = args.img_dir
    save_img_dir = args.save_dir
    print('img_dir=',img_dir)
    print('save_img_dir=',save_img_dir)
    print("=====blur parameter settings=====")
    do_blur = args.blur
    blur_type = args.blur_type
    blur_size = args.blur_size
    print('do_blur =',do_blur)
    print('blur_type=',blur_type)
    print('blur_size=',blur_size)
    print("=====flip parameter settings=====")
    do_flip = args.flip
    flip_type = args.flip_type
    print("=====hsv parameter settings=====")
    do_hsv = args.hsv
    numv = args.numv
    #do_blur = True
    #do_flip = True
    #blur_type = 2
    #blur_size = 7
    #flip_type = 1
    #img_dir = "C:/TLR/datasets/roi-original"
    #save_img_dir = "C:/TLR/datasets"
    pure_img_augmentation(False,#do blur
                          blur_type,
                          blur_size,
                          False, #do flip
                          flip_type,
                          True, #do hsv
                          numv,
                              img_dir,
                              save_img_dir
                             
                              )
    
        