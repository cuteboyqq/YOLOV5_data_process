#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 19:59:25 2023

@author: ali
"""
import random
import glob
import os
import cv2
from matplotlib import pyplot as plt


def Analysis_path(path):
    file = path.split(os.sep)[-1]
    file_name = file.split(".")[0]
    file_dir = os.path.dirname(path)
    print("file = ",file)
    print("file_name = ",file_name)
    print("file_dir = ",file_dir)
    return file,file_name,file_dir

def cutout(im, p=1.0):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    if random.random() < p:
        h, w = im.shape[:2]
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
        min_value = min(h,w)
        for s in scales:
            #mask_h = random.randint(1, int(h * s))  # create random masks
            #mask_w = random.randint(1, int(w * s))
            
            mask_h = random.randint(1, int(min_value * s))  # create random masks
            mask_w = random.randint(1, int(min_value * s))
    
            # box
            
            #xmin = max(0, random.randint(0, w) - mask_w // 2)
            xmin = max(0, random.randint(0, min_value) - mask_w // 2)
            ymin = xmin
            #ymin = max(0, random.randint(0, h) - mask_h // 2)
            #xmax = min(w, xmin + mask_w)
            xmax = min(min_value, xmin + mask_w)
            
            ymax = xmax
            #ymax = min(h, ymin + mask_h)
    
            # apply random color mask
            #im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]
            im[ymin:ymax, xmin:xmax] = list(zip(*im[ymin:ymax, xmin:xmax][::-1]))
            ##Rotation Method https://stackoverflow.com/questions/8421337/rotating-a-two-dimensional-array-in-python
            # return unobscured labels
            #if len(labels) and s > 0.03:
                #box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                #ioa = bbox_ioa(box, xywhn2xyxy(labels[:, 1:5], w, h))  # intersection over area
                #labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return im



if __name__ == "__main__":
    
    infer_images=True
    infer_video=False
    
    if infer_images:
        img_dir = "/home/ali/datasets/factory_data/2022-12-30-4cls-cropimg/crops_line/line"
        img_path_list = glob.glob(os.path.join(img_dir,"*.jpg"))
        
        save_dir = "/home/ali/datasets/factory_data/2022-12-30-4cls-cropimg/crops_line_cutout_ver2"
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        for i in range(len(img_path_list)):
            print("{}, {}".format(i,img_path_list[i]))
            
            im = cv2.imread(img_path_list[i])
            im_cutout = cutout(im)
            img_file = str(i)+'.jpg'
            img_path = os.path.join(save_dir,img_file)
            
            
            #cv2.imwrite(img_path, im_cutout)
            
            plt.imshow(im_cutout)
            plt.show()
            
            #im_cutout.numpy()
            cv2.imwrite(img_path, im_cutout)
            
            
    if infer_video:
        
        img_size = 640
        path = "/home/ali/factory_video/Produce_short.mp4"
        count = 1
        file,filename,file_dir = Analysis_path(path)
        print(file," ",filename," ",file_dir)
        save_folder_name =  filename + "_imgs"
        save_dir = os.path.join(file_dir,save_folder_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        skip_frame = 10
        vidcap = cv2.VideoCapture(path)
        success,image = vidcap.read()
        
        count = 0
        
        while True:
            if success:
                if count%skip_frame==0:
                    
                    #====extract video frame====
                    filename_ = filename + "_" + str(count) + ".jpg"
                    img_path = os.path.join(save_dir,filename_)
                    image = cv2.resize(image,(img_size, int(img_size*9/16) ))
                    
                    im_cutout = cutout(image)
                    cv2.imwrite(img_path,im_cutout)
                    print("save image complete",img_path)
                    
                    #cv2.imwrite(img_path,image)
                    #if yolo_infer_txt:
                        #=====Copy .txt file=======
                        #filename_txt_ = filename + "_" + str(count) + ".txt"
                        #txt_path = os.path.join(txt_dir,filename_txt_)
                        #if os.path.exists(txt_path):
                            #shutil.copy(txt_path,save_dir)
                        
                    #cv2.imwrite("/home/ali/datasets-old/TL4/frame%d.jpg" % count, image)     # save frame as JPEG file    
                    print('save frame ',count)
            #else:
                #break
            success,image = vidcap.read()
            #print('Read a new frame: ', success)
            count += 1
