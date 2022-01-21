#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 15:35:37 2021

@author: ali
"""

import cv2
import os
import shutil

'''
path = "/home/ali/datasets/train_video/NewYork_train/NewYork_train7.mp4"
vidcap = cv2.VideoCapture(path)
skip_frame = 15

txt_dir = "/home/ali/YOLOV5/runs/detect/NewYork_train7/labels"
class_path = "/home/ali/datasets/train_video/classes.txt"

yolo_infer_txt = True
'''


def Analysis_path(path):
    file = path.split("/")[-1]
    file_name = file.split(".")[0]
    file_dir = os.path.dirname(path)
    return file,file_name,file_dir

#c_file,c_file_name,c_file_dir = Analysis_path(class_path)

def video_extract_frame(path,skip_frame,txt_dir,class_path,yolo_infer_txt):
    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    count = 1
    file,filename,file_dir = Analysis_path(path)
    print(file," ",filename," ",file_dir)
    save_folder_name =  filename + "_imgs"
    save_dir = os.path.join(file_dir,save_folder_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    #Copy class.txt to save_dir
    shutil.copy(class_path,save_dir)
    
    while success:
        if count%skip_frame==0:
            #====extract video frame====
            filename_ = filename + "_" + str(count) + ".jpg"
            img_path = os.path.join(save_dir,filename_)
            
            cv2.imwrite(img_path,image)
            if yolo_infer_txt:
                #=====Copy .txt file=======
                filename_txt_ = filename + "_" + str(count) + ".txt"
                txt_path = os.path.join(txt_dir,filename_txt_)
                shutil.copy(txt_path,save_dir)        
            #cv2.imwrite("/home/ali/datasets-old/TL4/frame%d.jpg" % count, image)     # save frame as JPEG file    
            print('save frame ',count)
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

def get_args():
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-videopath','--video-path',help="input video path",default="/home/ali/datasets/train_video/NewYork_train/NewYork_train8.mp4")
    parser.add_argument('-skipf','--skip-f',type=int,help="number of skp frame",default=10)
    parser.add_argument('-yoloinfer','--yolo-infer',action='store_true',help="have yolo infer txt")
    parser.add_argument('-yolotxt','--yolo-txt',help="yolo infer label txt dir",default="/home/ali/YOLOV4/inference/NewYork_train8/labels")
    parser.add_argument('-classtxt','--class-txt',help="class.txt path",default="/home/ali/datasets/train_video/classes.txt")
    
    return parser.parse_args()
    
if __name__=="__main__":
    
    args=get_args()
    video_path = args.video_path
    skip_frame = args.skip_f
    yolo_txt_dir = args.yolo_txt
    class_path = args.class_txt
    yolo_infer = args.yolo_infer
    
    print("video_path =",video_path)
    print("skip_frame = ",skip_frame)
    print("yolo_txt_dir = ",yolo_txt_dir)
    print("class_path = ",class_path)
    print("yolo_infer = ",yolo_infer)
    
    video_extract_frame(video_path,skip_frame,yolo_txt_dir,class_path,yolo_infer)
    