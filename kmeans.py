#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 18:50:16 2023

@author: ali
"""

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import shutil

# Function to Extract features from the images
def image_feature(direc):
    model = InceptionV3(weights='imagenet', include_top=False)
    features = [];
    img_name = [];
    for i in tqdm(direc):
        fname='/home/ali/datasets/factory_data/2022-12-30-4cls-cropimg/crops_line_10/line'+'/'+i
        img=image.load_img(fname,target_size=(224,224))
        x = img_to_array(img)
        x=np.expand_dims(x,axis=0)
        x=preprocess_input(x)
        feat=model.predict(x)
        feat=feat.flatten()
        features.append(feat)
        img_name.append(i)
    return features,img_name


img_path=os.listdir('/home/ali/datasets/factory_data/2022-12-30-4cls-cropimg/crops_line_10/line')
img_features,img_name=image_feature(img_path)

#Creating Clusters
k = 4
clusters = KMeans(k, random_state = 80)
clusters.fit(img_features)


image_cluster = pd.DataFrame(img_name,columns=['image'])
image_cluster["clusterid"] = clusters.labels_
image_cluster # 0 denotes cat and 1 denotes dog


# Made folder to seperate images
if not os.path.exists('line1'):
    os.mkdir('line1')
if not os.path.exists('line2'):
    os.mkdir('line2')
if not os.path.exists('line3'):
    os.mkdir('line3')
if not os.path.exists('line4'):
    os.mkdir('line4')
if not os.path.exists('line5'):
    os.mkdir('line5')
if not os.path.exists('line6'):
    os.mkdir('line6')
if not os.path.exists('line7'):
    os.mkdir('line7')
if not os.path.exists('line8'):
    os.mkdir('line8')
# Images will be seperated according to cluster they belong
for i in range(len(image_cluster)):
    if image_cluster['clusterid'][i]==0:
        shutil.copy(os.path.join('/home/ali/datasets/factory_data/2022-12-30-4cls-cropimg/crops_line_10/line', image_cluster['image'][i]), 'line1')
    elif image_cluster['clusterid'][i]==1:
        shutil.copy(os.path.join('/home/ali/datasets/factory_data/2022-12-30-4cls-cropimg/crops_line_10/line', image_cluster['image'][i]), 'line2')
    elif image_cluster['clusterid'][i]==2:
        shutil.copy(os.path.join('/home/ali/datasets/factory_data/2022-12-30-4cls-cropimg/crops_line_10/line', image_cluster['image'][i]), 'line3')
    elif image_cluster['clusterid'][i]==3:
        shutil.copy(os.path.join('/home/ali/datasets/factory_data/2022-12-30-4cls-cropimg/crops_line_10/line', image_cluster['image'][i]), 'line4')
    elif image_cluster['clusterid'][i]==4:
        shutil.copy(os.path.join('/home/ali/datasets/factory_data/2022-12-30-4cls-cropimg/crops_line_10/line', image_cluster['image'][i]), 'line5')
    elif image_cluster['clusterid'][i]==5:
        shutil.copy(os.path.join('/home/ali/datasets/factory_data/2022-12-30-4cls-cropimg/crops_line_10/line', image_cluster['image'][i]), 'line6')
    elif image_cluster['clusterid'][i]==6:
        shutil.copy(os.path.join('/home/ali/datasets/factory_data/2022-12-30-4cls-cropimg/crops_line_10/line', image_cluster['image'][i]), 'line7')
    else:
        shutil.copy(os.path.join('/home/ali/datasets/factory_data/2022-12-30-4cls-cropimg/crops_line_10/line', image_cluster['image'][i]), 'line8')

