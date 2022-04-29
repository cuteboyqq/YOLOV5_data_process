# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 17:30:09 2021

@author: admin
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import torch
import torchvision
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import tqdm
import math
'''
============================================================================================
Enable train or not

TRAIN : enable/disable train
TRAIN_EPOCH : enable/disable train model for epoch times
DRAW_TRAIN_LOSS_EPOCH_GRAPH : enable/disable draw train loss at each epoch
ENABLE_VALIDATION : enable/disable validation at each epoch

============================================================================================
'''
date = '-20220420-' #set~~~~~~~~~~
c1,c2,c3,c4 = 2,4,5,7 #set~~~~
DO_TRAIN = True
if DO_TRAIN:
    TRAIN = True
    TRAIN_EPOCH = True
    DRAW_TRAIN_LOSS_EPOCH_GRAPH = True
    ENABLE_VALIDATION = True
else:
    TRAIN = False
    TRAIN_EPOCH = False
    DRAW_TRAIN_LOSS_EPOCH_GRAPH = False
    ENABLE_VALIDATION = False
'''
======================================================
define plot file name  at each train epoch
======================================================
'''
ch = str(c1)+'-'+ str(c2) +'-'+ str(c3) +'-'+ str(c4) #set~~~~~~~~~
PLOT_PRECISION_NAME = "avg_precision" + date + ch + ".png"
PLOT_RECALL_NAME = 'avg_recall' + date + ch + ".png"
PLOT_TRAIN_LOSS_NAME = 'Train_Loss_Epoch' + date + ch + ".png"
PLOT_VAL_LOSS_NAE = 'Val_Loss_Epoch' + date + ch + ".png"
PLOT_ACC_NAME = 'avg_acc' + date + ch + ".png"
'''
========================================================
'''



'''
==============================================================================
COLLECT_FP_IMAGES : enable/disable collect FP of testing dataset infer result
==============================================================================
'''
COLLECT_FP_IMAGES = False

'''
============================================================================================
Train parameter settings

nums_epoch : num of train iterations
NUM_CLASS : num of train labels
BATCH_SIZE : num of train batch size
IMAGE_SIZE : train image size
PATH : save model path
CONFUSION_MATRIX_MODEL :  model path for generate confusion matrix, here is PATH
============================================================================================
'''

nums_epoch = 50
NUM_CLASS = 8
BATCH_SIZE = 300
IMAGE_SIZE = 32
TRAIN_DATA_DIR = '/home/ali/TLR/datasets/roi'
VAL_DATA_DIR = '/home/ali/TLR/datasets/roi-test'
PATH = '/home/ali/TLR/model/TLR_ResNet18' + date + '-Size32-' + ch + '.pt' #set~~~~~
CONFUSION_MATRIX_MODEL = PATH
CM_FILENAME = 'confusion_matrix' + date + ch + '.png' #set~~~~~
#modelPath = PATH #for confusion matrix model input


'''
==============================================================================================================
Enable inference

INFERENCE: enable/disable inference test datasets
GET_INFER_ACCURACY : enable/disable calculate inference reacll
CONFUSION_MATRIX : enable/disable use pytorch to do testing, data is inference with batch size images,
                    and generate confusion matrix, precision, recall
==============================================================================================================
'''
INFERENCE = True
INFERENCE_MODEL = PATH
PREDICT_RESULT_IMG = '/home/ali/TLR/inference/TLR_ResNet18' + date + 'Size32' + ch + '_result' #set~~~~~
INFERENCE_DATA_DIR = VAL_DATA_DIR
GET_INFER_ACCURACY = True
CONFUSION_MATRIX = True

'''
============================================================================================================
Congratulations, you have complete parameter settings, you can start train/validation/test model
============================================================================================================
'''
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import os
import random
import shutil
import time
import json
import warnings




def validate(test_loader, model, criterion, y_pred, y_true):
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(test_loader):
            #if args.gpu is not None:
            #images = images.cuda('cuda', non_blocking=True)
            if torch.cuda.is_available():
                images = images.cuda('cuda', non_blocking=True)
                target = target.cuda('cuda', non_blocking=True)
            else:
                images = images
                target = target

            #output = model(images)
            output = F.softmax(model(images)).data
            _, preds = torch.max(output, 1)                            # preds是預測結果
            loss = criterion(output, target)
            #print("loss =", loss)
            y_pred.extend(preds.view(-1).detach().cpu().numpy())       # 將preds預測結果detach出來，並轉成numpy格式       
            y_true.extend(target.view(-1).detach().cpu().numpy())      # target是ground-truth的label
            
    return y_pred, y_true, loss



'''
==============================================================================================================================================

Train model

======================================================================================================================================
'''
if TRAIN:
    '''
    =====================================================================================
    function:torchvision.datasets.ImageFolder
    
    this function can get train image by put the  directory of image
    inside the folder, you need to classify images by folder
    
    for example:
    folder directory: D:/Traffic_Light_Classify/
    under D:/Traffic_Light_Classify/ , create some sub-folders that put images in sub-folders
    D:/Traffic_Light_Classify/Red   : put red images (.png or .jpg)
    D:/Traffic_Light_Classify/Green : put green images (.png or .jpg)
    D:/Traffic_Light_Classify/Yellow :put yellow images (.png or .jpg)
    ...
    ...
    D:/Traffic_Light_Classify/Background : put background images (.png or .jpg)
    =======================================================================================
    '''
    size = (IMAGE_SIZE,IMAGE_SIZE)
    img_data = torchvision.datasets.ImageFolder(TRAIN_DATA_DIR,
                                                transform=transforms.Compose([
                                                    transforms.Resize(size),
                                                    #transforms.RandomHorizontalFlip(),
                                                    #transforms.Scale(64),
                                                    transforms.CenterCrop(size),
                                                 
                                                    transforms.ToTensor()
                                                    ])
                                                )
    
    print(len(img_data))
    '''
    ============================================================================
    function : torch.utils.data.DataLoader
    
    load the images to tensor
    
    ============================================================================
    '''
    data_loader = torch.utils.data.DataLoader(img_data, batch_size=BATCH_SIZE,shuffle=True,drop_last=False)
    print(len(data_loader))
    
    
    classes = ('GreenLeft', 'GreenRight', 'GreenStraight','RedLeft','RedRight','YellowLeft','YellowRight','others')
    # 显示一张图片
    def imshow(img):
        img = img / 2 + 0.5   # 逆归一化
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    
    
    # 任意地拿到一些图片
    dataiter = iter(data_loader)
    images, labels = dataiter.next()
    
    # 显示图片
    imshow(torchvision.utils.make_grid(images))
    # 显示类标
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    
    
    
    def show_batch(imgs):
        grid = utils.make_grid(imgs,nrow=5)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.title('Batch from dataloader')
    
    
    for i, (batch_x, batch_y) in enumerate(data_loader):
        if(i<6):
            print(i, batch_x.size(), batch_y.size())
    
            show_batch(batch_x)
            plt.axis('off')
            plt.show()
            
    
    for i, data in enumerate(data_loader):
      img,label=data
      print(i," : ",label)
    
    
    import torch.nn as nn
    import torch.nn.functional as F
    
   #定义残差块ResBlock
    class ResBlock(nn.Module):
        def __init__(self, inchannel, outchannel, stride=1):
            super(ResBlock, self).__init__()
            #这里定义了残差块内连续的2个卷积层
            self.left = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(inplace=True),
                nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            self.shortcut = nn.Sequential()
            if stride != 1 or inchannel != outchannel:
                #shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
                self.shortcut = nn.Sequential(
                    nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(outchannel)
                )
                
        def forward(self, x):
            out = self.left(x)
            #将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
            out = out + self.shortcut(x)
            out = F.relu(out)
            
            return out
    
    class ResNet(nn.Module):
        def __init__(self, ResBlock, num_classes=NUM_CLASS):
            super(ResNet, self).__init__()
            self.inchannel = 64
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
            self.layer1 = self.make_layer(ResBlock, c1, 2, stride=1)#16
            self.layer2 = self.make_layer(ResBlock, c2, 2, stride=2)#32
            self.layer3 = self.make_layer(ResBlock, c3, 2, stride=2)#64        
            self.layer4 = self.make_layer(ResBlock, c4, 2, stride=2)#128        
            self.fc = nn.Linear(c4, num_classes)#512 for 64*64,128 for 32*32
        #这个函数主要是用来，重复同一个残差块    
        def make_layer(self, block, channels, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for stride in strides:
                layers.append(block(self.inchannel, channels, stride))
                self.inchannel = channels
            return nn.Sequential(*layers)
        
        def forward(self, x):
            #在这里，整个ResNet18的结构就很清晰了
            out = self.conv1(x)
            #print(out.shape)
            out = self.layer1(out)
            #print(out.shape)
            out = self.layer2(out)
            #print(out.shape)
            out = self.layer3(out)
            #print(out.shape)
            out = self.layer4(out)
            #print(out.shape)
            out = F.avg_pool2d(out, 4)
            #print(out.shape)
            out = out.view(out.size(0), -1)
            #print(out.shape)
            out = self.fc(out)
            #print(out.shape)
            return out
    
    net = ResNet(ResBlock)
    if torch.cuda.is_available():
        net.cuda()
    print(net)
    
    params = list(net.parameters())
    print(len(params))
    
    for name, parameters in net.named_parameters():
        print(f'{name}: {parameters.size()}')
        
        
    
        
    if TRAIN_EPOCH:
        import torch.optim as optim
        '''loss function'''
        criterion = nn.CrossEntropyLoss()
        ''' optimizer method '''
        optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
            
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _lowest_loss = 1000.0
        #PATH = "/home/ali/TLR/model/TLR_ResNet18-2022-04-08.pt"
        '''
        =============================================================================
        
        Start training model for nums_epoch times
        
        =============================================================================
        '''
        loss_history = []
        epochs = []
        avg_precision_list = []
        avg_recall_list = []
        val_loss_list = []
        avg_acc_list = []
        
        val_precision_folder_dir = r'/home/ali/TLR/plot_graph/precision/val'
        val_recall_folder_dir = r'/home/ali/TLR/plot_graph/recall/val'
        train_loss_dir = r'/home/ali/TLR/plot_graph/loss/train'
        val_loss_dir = r'/home/ali/TLR/plot_graph/loss/val'
        val_acc_dir = r'/home/ali/TLR/plot_graph/acc/val'
        
        if not os.path.exists(val_precision_folder_dir):
            os.makedirs(val_precision_folder_dir)
            
        if not os.path.exists(val_recall_folder_dir):
            os.makedirs(val_recall_folder_dir)
            
        if not os.path.exists(train_loss_dir):
            os.makedirs(train_loss_dir)
            
        if not os.path.exists(val_loss_dir):
            os.makedirs(val_loss_dir)
        
        if not os.path.exists(val_acc_dir):
            os.makedirs(val_acc_dir)
        
        sm_pre = 0.0
        sm_recall = 0.0
        sm_acc = 0.0
        save_model = 1
        for epoch in range(nums_epoch):
            total_loss = 0.0
            tot_loss = 0.0 
            _loss = 0.0
            train_preds = []
            train_trues = []
            y_pred = []   #保存預測label
            y_true = []   #保存實際label
            
            for i, (inputs, labels) in enumerate(data_loader, 0):
                '''get batch images and corresponding labels'''
                inputs, labels = inputs.to(device), labels.to(device)
                '''initial optimizer to zeros'''
                optimizer.zero_grad()
                ''' put batch images to convolution neural network '''
                outputs = net(inputs)
                """calculate loss by loss function"""
                loss = criterion(outputs, labels)
                '''after calculate loss, do back propogation'''
                loss.backward()
                '''optimize weight and bais'''
                optimizer.step()
                  
                _loss += loss.item()
                tot_loss += loss.data
                total_loss += loss.item()
                train_outputs = outputs.argmax(dim=1)
                
                train_preds.extend(train_outputs.detach().cpu().numpy())
                train_trues.extend(labels.detach().cpu().numpy())
                '''
                =======================================================
                
                After some epochs , Save the model which loss is lowest 
                (Noted, not save model currently, just to show loss info.)
                
                =======================================================
                '''
                if i % 6 == 0 and i > 0:  # 每3步打印一次损失值
                    print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, _loss / 1))
                    #if epoch > 0:
                    '''    
                    if _loss < _lowest_loss:
                        _lowest_loss = _loss
                        print('Start save model !')
                        torch.save(net, PATH)
                        print('save model complete with loss : %.3f' %(_loss))
                    '''
                    _loss = 0.0
            '''
            ==========================================================
            Save model if loss is the smallest at each epoch
            ==========================================================
            '''
            if tot_loss < _lowest_loss:
                save_model = epoch+1
                _lowest_loss = tot_loss
                print('Start save model !')
                torch.save(net, PATH)
                print('save model complete with loss : %.3f' %(tot_loss))
            epochs.extend([epoch+1])
            '''
            ==========================================================
            plot train loss at each epochs
            ==========================================================
            '''
            if DRAW_TRAIN_LOSS_EPOCH_GRAPH:
                loss_history.extend([int(total_loss)])
                #epochs.extend([epoch+1])
                print(epochs)
                print(loss_history)
                plt.figure(figsize = (int(nums_epoch/3),9))
                for a,b in zip(epochs, loss_history): 
                    plt.text(a, b, str(b))
                plt.plot(epochs,loss_history)
                plt.xlabel('epochs')
                plt.ylabel('loss_history')
                plt.title("loss at each epoch")
                save_path = os.path.join(train_loss_dir,PLOT_TRAIN_LOSS_NAME)
                plt.savefig(save_path)
                plt.show()
                
            '''
            ======================================================
            Calculate at each Training Epochs :
            1. precision
            2. recall
            3. f1-score
            ======================================================
            '''
            
            
            sklearn_accuracy = accuracy_score(train_trues, train_preds) 
            sklearn_precision = precision_score(train_trues, train_preds, average='micro')
            sklearn_recall = recall_score(train_trues, train_preds, average='micro')
            sklearn_f1 = f1_score(train_trues, train_preds, average='micro')
            print("[sklearn_metrics] Epoch:{} loss:{:.4f} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}"
                  .format(epoch, tot_loss, sklearn_accuracy, sklearn_precision, sklearn_recall, sklearn_f1))
            '''
            =======================================================
            '''
            
            '''
            =======================================================
            do validation at each epoch 2022-04-14
            =======================================================
            '''
            modelPath = PATH
            if os.path.exists(modelPath):
                MODEL_EXIST = True
            else:
                MODEL_EXIST = False
            if ENABLE_VALIDATION and MODEL_EXIST:
                
                IMAGE_SIZE = 32
                #modelPath = r"/home/ali/TLR/model/TLR_ResNet18-2022-04-14-Size32-4-8-16-32.pt"
                size = (IMAGE_SIZE,IMAGE_SIZE)
                img_test_data = torchvision.datasets.ImageFolder(VAL_DATA_DIR,
                                                            transform=transforms.Compose([
                                                                transforms.Resize(size),
                                                                #transforms.RandomHorizontalFlip(),
                                                                #transforms.Scale(64),
                                                                transforms.CenterCrop(size),
                                                             
                                                                transforms.ToTensor()
                                                                ])
                                                            )
                
                print(len(img_test_data))
                '''
                ============================================================================
                function : torch.utils.data.DataLoader
                
                load the images to tensor
                
                ============================================================================
                '''
                BATCH_SIZE_VAL = BATCH_SIZE
                test_loader = torch.utils.data.DataLoader(img_test_data, batch_size=BATCH_SIZE_VAL,shuffle=False,drop_last=False)
                print(len(test_loader))
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                #device = torch.device('cpu')
                model = torch.load(modelPath).to(device)
                criterion = nn.CrossEntropyLoss()
                
                y_pred, y_true, val_loss = validate(test_loader, model, criterion, y_pred,y_true)
    
                # 製作混淆矩陣
                cf_matrix = confusion_matrix(y_true, y_pred)                                # https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-in-pytorch-38d06a7f04b7
                # 計算每個class的accuracy
                per_cls_acc = cf_matrix.diagonal()/cf_matrix.sum(axis=0)                    # https://stackoverflow.com/a/53824126/13369757
                class_names = ['GreenLeft', 'GreenRight', 'GreenStraight','RedLeft','RedRight','YellowLeft','YellowRight','others']
                print(class_names)
                print(per_cls_acc)                                                          #顯示每個class的Accuracy
                #print("Plot confusion matrix")
                '''
                # 開始繪製混淆矩陣並存檔
                df_cm = pd.DataFrame(cf_matrix, class_names, class_names)    
                #ax1 = plt.subplot(1,2,1)
                plt.figure(figsize = (9,6))
                sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
                plt.xlabel("prediction")
                plt.ylabel("label (ground truth)")
                plt.savefig("confusion_matrix-20220414-4-8-16-32.png")
                '''
                '''
                ============================================================================
                calculate test datasets:
                    
                precision
                recall
                =============================================================================
                '''
                TP = cf_matrix.diagonal()
                FP = cf_matrix.sum(0) - TP
                FN = cf_matrix.sum(1) - TP
                
                #avg_precision_list = []
                #avg_recall_list = []
                    
                precision = TP / (TP + FP + 1e-12) 
                recall =   TP / (TP + FN + 1e-12)
                acc = [TP.sum()]*len(TP) / ([TP.sum()]*len(TP) + FP + FN)
                #avg_acc = per_cls_acc.mean()
                avg_precision = precision.mean()
                avg_recall = recall.mean()
                avg_acc = acc.mean()
                
                '''add avg_pre, avg_recall, val_loss to list'''
                avg_precision_list.extend([avg_precision])
                avg_recall_list.extend([avg_recall])
                val_loss_list.extend(val_loss.view(-1).detach().cpu().numpy())
                avg_acc_list.extend([avg_acc])
                '''
                ==========================================================================
                plot precision and recall and loss at each epoch 
                on validation datasets
                ==========================================================================
                '''
                #==============plot precision at each epoch======================= 
                print("plot precision at each epoch :")
                print(epochs)
                print(avg_precision_list)
                num = 3
                
                plt.figure(figsize = (int(nums_epoch/3),9))
                for a,b in zip(epochs, avg_precision_list):
                    txt = float(int(b*10000))/float(100.0)
                    if save_model==a:
                        plt.text(a, b, str(txt)+'%,sm')
                        sm_pre = b
                    if a%num == 0 or a==len(avg_precision_list):
                        plt.text(a, b, str(txt)+'%')
                        
                plt.plot(epochs,avg_precision_list)
                plt.xlabel('epochs')
                plt.ylabel('avg_precision')
                title_txt = "avg_precision at each epoch, sm: epoch=" + str(a) + " pre=" + str(sm_pre)
                plt.title(title_txt)
                save_path = os.path.join(val_precision_folder_dir,PLOT_PRECISION_NAME)
                plt.savefig(save_path)
                plt.show()
                #==================================================================
                #==============plot recall at each epoch===========================
                print("plot recall at each epoch :")
                print(epochs)
                print(avg_recall_list)
                plt.figure(figsize = (int(nums_epoch/3),9))
                
                for a,b in zip(epochs, avg_recall_list):
                    txt = float(int(b*10000))/float(100.0)
                    if save_model==a:
                        plt.text(a, b, str(txt)+'%,sm')
                        sm_recall = b
                    if a%num == 0 or a==len(avg_recall_list):
                        plt.text(a, b, str(txt)+'%')
                        
                plt.plot(epochs,avg_recall_list)
                plt.xlabel('epochs')
                plt.ylabel('avg_recall')
                title_txt = "avg_recall at each epoch, sm: epoch=" + str(a) + " recall=" + str(sm_recall)
                plt.title(title_txt)
                save_path = os.path.join(val_recall_folder_dir,PLOT_RECALL_NAME)
                plt.savefig(save_path)
                plt.show()
                
                #==================================================================
                #==============plot val loss at each epoch=========================
                
                print("plot val_loss at each epoch :")
                print(epochs)
                print(val_loss_list)
                plt.figure(figsize = (int(nums_epoch/3),9))
                for a,b in zip(epochs, val_loss_list): 
                    plt.text(a, b, str(b))
                plt.plot(epochs,val_loss_list)
                plt.xlabel('epochs')
                plt.ylabel('val_loss')
                plt.title("val_loss at each epoch")
                save_path = os.path.join(val_loss_dir,PLOT_VAL_LOSS_NAE)
                plt.savefig(save_path)
                plt.show()
                #=================================================================
                #============plot acc at each epoch===============================
                print("plot acc at each epoch :")
                print(epochs)
                print(avg_acc_list)
                plt.figure(figsize = (int(nums_epoch/3),9))
                for a,b in zip(epochs, avg_acc_list):
                    if not math.isnan(b):   
                        txt = float(int(b*10000))/float(100.0)
                    else:
                        txt = b
                        
                    if save_model==a:
                        plt.text(a, b, str(txt)+'%,sm')
                        sm_acc = b
                    if a%num == 0 or a==len(avg_recall_list):
                        plt.text(a, b, str(txt)+'%')
                    
                plt.plot(epochs,avg_acc_list)
                plt.xlabel('epochs')
                plt.ylabel('avg_acc')
                title_txt = "avg_accuracy at each epoch, sm: epoch=" + str(a) + " avg_acc=" + str(sm_acc)
                plt.title(title_txt)
                save_path = os.path.join(val_acc_dir,PLOT_ACC_NAME)
                plt.savefig(save_path)
                plt.show()
                #===================================================================
                print("TP: {}, FP: {}, FN: {}".format(TP,FP,FN))
                print("precision = TP/(TP + FP) :")
                print("{}".format(precision))
                print("recall = TP/(TP + FN) :")
                print("{}".format(recall))
                print("avg prcision = {}".format(avg_precision))
                print("avg recall = {}".format(avg_recall))
            '''
            ========================================================
            ========================================================
            '''
        print('Finished Training')
        #PATH = "D:/YOLOX-main/assets/TLR_model/TLR.pt"
        #torch.save(net.state_dict(), PATH)
        #print('Start save model')
        #torch.save(net, PATH)
        #print('save model complete')
'''
=================================================================================================
infernece function
input: imagePath
the image dir that we want to inference

input: modelPath
the path of the .th model 
=================================================================================================
''' 
#========================================================================================== 

from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
'''
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5) #64
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(10000, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 3)
    
  def forward(self, x):
    print("Inference Start forward``````````````````")
    x = self.pool(F.relu(self.conv1(x)))
    print(x.shape)
    x = self.pool(F.relu(self.conv2(x)))
    print(x.shape)
    x = x.view(1,-1)
    print(x.shape)
    #x = x.view(x.size(0),-1)
    x = F.relu(self.fc1(x))
    print(x.shape)
    x = F.relu(self.fc2(x))
    print(x.shape)
    x = self.fc3(x)
    print(x.shape)
    return x


net = Net()
'''
 #定义残差块ResBlock
class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        #这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            #shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            
    def forward(self, x):
        out = self.left(x)
        #将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, ResBlock, num_classes=NUM_CLASS):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResBlock, c1, 2, stride=1)#16
        self.layer2 = self.make_layer(ResBlock, c2, 2, stride=2)#32
        self.layer3 = self.make_layer(ResBlock, c3, 2, stride=2)   #64     
        self.layer4 = self.make_layer(ResBlock, c4, 2, stride=2)     #128   
        self.fc = nn.Linear(c4, num_classes)#128
    #这个函数主要是用来，重复同一个残差块    
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        #在这里，整个ResNet18的结构就很清晰了
        out = self.conv1(x)
        #print(out.shape)
        out = self.layer1(out)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = self.layer3(out)
        #print(out.shape)
        out = self.layer4(out)
        #print(out.shape)
        out = F.avg_pool2d(out, 4)
        #print(out.shape)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.fc(out)
        #print(out.shape)
        return out
  
  
net = ResNet(ResBlock)
#if torch.cuda.is_available():
    #net.cuda()
print(net)
import sys
from PIL import Image
from urllib.request import urlopen
import cv2
import glob
'''
=======================================================================================================================================
function : inference

infernece images
input imagedir:
    the image folder dir
modelPath :
    the model path
=========================================================================================================================================
'''
def inference(imagedir, modelPath,pred_dir):
    with open(r"/home/ali/TLR/classes.txt", "r") as f:
        classes = f.read().split("\n")
   
        
    inputSize = (32,32)

    data_transforms_test = transforms.Compose([
                        #transforms.ToPILImage(),
                        transforms.Resize(inputSize),
                        transforms.CenterCrop(inputSize),
                        transforms.ToTensor()
                        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = torch.load(modelPath).to(device)
    model.eval()
    #image = Image.open(urlopen(imagePath)).convert('RGB')
    #image = Image.open(imagePath).convert('RGB')
    #image = cv2.imread(imagePath)
    search_img_dir = imagedir + "/**/*.jpg"
    img_list = glob.iglob(search_img_dir)
    img_count = 0
    y_pred = []   #保存預測label
    y_true = []   #保存實際label
    for img_path in img_list:
        GT = os.path.basename(os.path.dirname(img_path))
        img = cv2.imread(img_path) # opencv開啟的是BRG
        #cv2.imshow("OpenCV",img)
        image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        #image.show()
        #cv2.waitKey()
    
    
        image_tensor = data_transforms_test(image).float()
        image_tensor = image_tensor.unsqueeze_(0).to(device)
        output = F.softmax(model(image_tensor)).data.cpu().numpy()[0]
        prediction = classes[np.argmax(output)]
        pre_label = np.argmax(output)
        score = max(output)
        print(score, prediction," ",GT)
        result_img_name = prediction + "_" + str(score)
        
        '''
        =================================================================================
        '''
        #y_pred.extend(str(prediction))
        #y_true.extend(str(GT))
        #print(len(y_pred))
        #print(len(y_true))
        '''
        =============================================================================================================
        Save inference images to folder
        ==============================================================================================================
        '''
        
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        pred_label_dir = pred_dir +"/" +prediction
        if not os.path.exists(pred_label_dir):
            os.makedirs(pred_label_dir)
        """
        image name format: [GroundTruth]_[Predict]_[Score]_[img_count].png
        """
        save_pred_img_path = pred_label_dir + "/"+GT+"_"+prediction+"_" + str(score) + "_" + str(img_count)+".png"
        cv2.imwrite(save_pred_img_path,img)
        img_count = img_count + 1
    '''
    ===============================================================================================================
    
    class_names = ['GreenLeft', 'GreenRight', 'GreenStraight','RedLeft','RedRight','YellowLeft','YellowRight','others']
    # 製作混淆矩陣
    cf_matrix = confusion_matrix(y_true, y_pred,labels=class_names)                                # https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-in-pytorch-38d06a7f04b7
    # 計算每個class的accuracy
    per_cls_acc = cf_matrix.diagonal()/cf_matrix.sum(axis=0)                    # https://stackoverflow.com/a/53824126/13369757
    
    print(class_names)
    print(per_cls_acc)                                                          #顯示每個class的Accuracy
    print("Plot confusion matrix")
    
    # 開始繪製混淆矩陣並存檔
    df_cm = pd.DataFrame(cf_matrix, class_names, class_names)    
    #ax1 = plt.subplot(1,2,1)
    plt.figure(figsize = (9,6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("prediction")
    plt.ylabel("label (ground truth)")
    plt.savefig("confusion_matrix.png")
    '''
#========================Do Inference images=====================================================================

if INFERENCE:
    imagedir = INFERENCE_DATA_DIR
    #imagePath = "D:\\YOLOX\\assets\\TL_Images\\off\\Red_frame2214_roi_2.png"
    modelPath = INFERENCE_MODEL
    pred_dir = PREDICT_RESULT_IMG
    inference(imagedir, modelPath,pred_dir)
    
    
#def Analysis_Image_Path(img_path):
    
        
import os
import glob

'''
===================================================================
function : Analysis_Image_Path
Analysis inference image by image name
 image name format: [GroundTruth]_[Predict]_[Score]_[img_count].png
======================================================================
'''
def Analysis_Image_Path(img_path):
    #GT = os.path.basename(os.path.dirname(img_path))
    img_name = os.path.basename(img_path)
    GT = img_name.split("_")[0]
    predict = img_name.split("_")[1]
    score = img_name.split("_")[2]
    
    return GT,predict,score
   
def Calculate_Inference_Accuracy(imagedir):
    search_img_dir = imagedir + "/**/*.png"
    img_list = glob.iglob(search_img_dir)
    print('img_list:',img_list)
    redleft_correct = 0
    redleft_wrong = 0
    redleft_accuracy = 0.0
    greenleft_correct = 0
    greenleft_wrong = 0
    greenleft_accuracy = 0.0
    others_acc = 0.0
    others_co = 0
    others_wr = 0
    yellowleft_acc,yellowleft_co,yellowleft_wr = 0.0,0,0
    greenStr_acc,greenStr_co,greenStr_wr = 0.0,0,0
    rright_acc,rright_co,rright_wr = 0.0,0,0
    gright_acc,gright_co,gright_wr = 0.0,0,0
    yright_acc,yright_co,yright_wr = 0.0,0,0
    for img_path in img_list:
        #print(img_path)
        GT,predict,score = Analysis_Image_Path(img_path)
        
        if GT == "RedLeft":
            if predict == "RedLeft":
                redleft_correct+=1
            else:
                redleft_wrong+=1
        elif GT == "GreenLeft":
            if predict =="GreenLeft":
                greenleft_correct+=1
            else:
                greenleft_wrong+=1
        elif GT == "others":
            if predict == "others":
                others_co+=1
            else:
                others_wr+=1
        elif GT == "YellowLeft":
            if predict == "YellowLeft":
                yellowleft_co+=1
            else:
                yellowleft_wr+=1
                
        elif GT == "GreenStraight":
            if predict == "GreenStraight":
                greenStr_co+=1
            else:
                greenStr_wr+=1
        elif GT == "RedRight":
            if predict == "RedRight":
                rright_co+=1
            else:
                rright_wr+=1
        elif GT == "GreenRight":
            if predict == "GreenRight":
                gright_co+=1
            else:
                gright_wr+=1
        elif GT == "YellowRight":
            if predict == "YellowRight":
                yright_co+=1
            else:
                yright_wr+=1
    
    redleft_accuracy = int(float(redleft_correct / (redleft_correct+redleft_wrong)) * 100)
    greenleft_accuracy = int(float(greenleft_correct / (greenleft_correct+greenleft_wrong)) *100)
    others_acc = int(float(others_co/(others_co+others_wr)) *100)
    yellowleft_acc = int(float(yellowleft_co/(yellowleft_co+yellowleft_wr)) *100)
    greenStr_acc = int(float(greenStr_co/(greenStr_co+greenStr_wr)) *100)
    rright_acc = int(float(rright_co/(rright_co+rright_wr)) *100)
    gright_acc = int(float(gright_co/(gright_co+gright_wr)) *100)
    yright_acc = int(float(yright_co/float(yright_co+yright_wr)) *100)
    print("redleft total = ",(redleft_correct+redleft_wrong),"correct:",redleft_correct,"wrong:",redleft_wrong,"acc:",redleft_accuracy,"%")
    print("greenleft total = ",(greenleft_correct+greenleft_wrong),"correct:",greenleft_correct,"wrong:",greenleft_wrong,"acc:",greenleft_accuracy,"%")
    print("yellowleft total = ",(yellowleft_co+yellowleft_wr),"correct:",yellowleft_co,"wrong:",yellowleft_wr,"acc:",yellowleft_acc,"%")
    print("others total = ",(others_co+others_wr),"correct:",others_co,"wrong:",others_wr,"acc:",others_acc,"%")
    print("greenStraight total = ",(greenStr_co+greenStr_wr),"correct:",greenStr_co,"wrong:",greenStr_wr,"acc:",greenStr_acc,"%")
    print("red right total = ",(rright_co+ rright_wr),"correct:",rright_co,"wrong:",rright_wr,"acc:",rright_acc,"%")
    print("green right total = ",(gright_co+ gright_wr),"correct:",gright_co,"wrong:",gright_wr,"acc:",gright_acc,"%")
    print("yellow right total = ",(yright_co+ yright_wr),"correct:",yright_co,"wrong:",yright_wr,"acc:",yright_acc,"%")
    
    
'''
 ===========================================================================================================
 function: Calculate_Inference_Accuracy

 Get inference accuracy
 input imagedir:
     the infer image folder dir
 
 ===========================================================================================================
'''   
if GET_INFER_ACCURACY:  
    
    imagedir = PREDICT_RESULT_IMG
    Calculate_Inference_Accuracy(imagedir)
    
#=========================================================================================================================
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import os
import random
import shutil
import time
import json
import warnings
'''
y_pred = []   #保存預測label
y_true = []   #保存實際label

def validate(test_loader, model, criterion, y_pred, y_true):
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(test_loader):
            #if args.gpu is not None:
            #images = images.cuda('cuda', non_blocking=True)
            if torch.cuda.is_available():
                images = images.cuda('cuda', non_blocking=True)
                target = target.cuda('cuda', non_blocking=True)
            else:
                images = images
                target = target

            #output = model(images)
            output = F.softmax(model(images)).data
            _, preds = torch.max(output, 1)                            # preds是預測結果
            loss = criterion(output, target)
            
            y_pred.extend(preds.view(-1).detach().cpu().numpy())       # 將preds預測結果detach出來，並轉成numpy格式       
            y_true.extend(target.view(-1).detach().cpu().numpy())      # target是ground-truth的label
        
    return y_pred, y_true
        
'''

if CONFUSION_MATRIX:
    IMAGE_SIZE = 32
    modelPath = CONFUSION_MATRIX_MODEL
    print(modelPath)
    size = (IMAGE_SIZE,IMAGE_SIZE)
    img_test_data = torchvision.datasets.ImageFolder(VAL_DATA_DIR,
                                                transform=transforms.Compose([
                                                    transforms.Resize(size),
                                                    #transforms.RandomHorizontalFlip(),
                                                    #transforms.Scale(64),
                                                    transforms.CenterCrop(size),
                                                 
                                                    transforms.ToTensor()
                                                    ])
                                                )
    
    print(len(img_test_data))
    '''
    ============================================================================
    function : torch.utils.data.DataLoader
    
    load the images to tensor
    
    ============================================================================
    '''
    BATCH_SIZE = 300
    test_loader = torch.utils.data.DataLoader(img_test_data, batch_size=BATCH_SIZE,shuffle=False,drop_last=False)
    print(len(test_loader))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    model = torch.load(modelPath).to(device)
    criterion = nn.CrossEntropyLoss()
    y_pred = []   #保存預測label
    y_true = []   #保存實際label
    y_pred, y_true, val_loss = validate(test_loader, model, criterion, y_pred,y_true)

    # 製作混淆矩陣
    cf_matrix = confusion_matrix(y_true, y_pred)                                # https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-in-pytorch-38d06a7f04b7
    # 計算每個class的accuracy
    per_cls_acc = cf_matrix.diagonal()/cf_matrix.sum(axis=0)                    # https://stackoverflow.com/a/53824126/13369757
    class_names = ['GreenLeft', 'GreenRight', 'GreenStraight','RedLeft','RedRight','YellowLeft','YellowRight','others']
    print(class_names)
    print(per_cls_acc)                                                          #顯示每個class的Accuracy
    print("Plot confusion matrix")
    
    # 開始繪製混淆矩陣並存檔
    df_cm = pd.DataFrame(cf_matrix, class_names, class_names)    
    #ax1 = plt.subplot(1,2,1)
    plt.figure(figsize = (9,6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("prediction")
    plt.ylabel("label (ground truth)")
    if not os.path.exists("/home/ali/TLR/confusion_matrix"):
        os.makedirs("/home/ali/TLR/confusion_matrix")
    save_cm_path = "/home/ali/TLR/confusion_matrix/"+ CM_FILENAME
    plt.savefig(save_cm_path)
    
    '''
    ============================================================================
    calculate test datasets:
        
    precision
    recall
    =============================================================================
    '''
    TP = cf_matrix.diagonal()
    FP = cf_matrix.sum(0) - TP
    FN = cf_matrix.sum(1) - TP
    
    #precision = []
    #recall = []
        
    precision = TP / (TP + FP + 1e-12) 
    recall =   TP / (TP + FN + 1e-12)
    avg_precision = precision.mean()
    avg_recall = recall.mean()
    
    
    print("TP: {}, FP: {}, FN: {}".format(TP,FP,FN))
    print("precision = TP/(TP + FP) :")
    print("{}".format(precision))
    print("recall = TP/(TP + FN) :")
    print("{}".format(recall))
    print("avg prcision = {}".format(avg_precision))
    print("avg recall = {}".format(avg_recall))
    
    
    '''
    ============================================================================
    plot precision
    plot recall
    ============================================================================
    '''
    DRAW_PRECISION_RECALL=False
    if DRAW_PRECISION_RECALL:
    
        # Import Library
    
        import numpy as np 
        #import matplotlib.pyplot as plt 
        
        # Define Data
        pre_dict = {}
        
        for i in range(len(class_names)):
            pre_dict[class_names[i]] = precision[i]
        
        keys = pre_dict.keys()
        values = pre_dict.values()
        plt.bar(keys, values)
        
        precision = precision
        df = pd.DataFrame({"prcision": precision,
                        "class_name": class_names})
        #ax2 = plt.subplot(1,2,2)
        #s1 = sns.barplot(x = 'class_name', y = 'prcision', data = df, color = 'red', ax=ax2)
        
        
        #precision = [115, 215, 250, 200]
        #recall = [114, 230, 510, 370]
          
        n=8
        r = np.arange(n)
        width = 0.25
          
          
        plt.bar(r, precision, color = 'b',
                width = width, edgecolor = 'black',
                label='precision')
        plt.bar(r + width, recall, color = 'g',
                width = width, edgecolor = 'black',
                label='recall')
          
        plt.xlabel("class names")
        plt.ylabel("percentage")
        plt.title("precision and recall")
          
        # plt.grid(linestyle='--')
        plt.xticks(r + width/2,class_names)
        plt.legend()
         
        plt.show()
        
        
        
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

import glob
import os
import shutil
import tqdm
if COLLECT_FP_IMAGES:
    
    def Analysis_img_path(img_path):
        img = os.path.basename(img_path)
        GT = img.split("_")[0]
        predict = img.split("_")[1]
        return GT, predict
    
    save_dir = "/home/ali/TLR/FP_datasets-2022-04-15-finetune"
    if not os.path.exists(save_dir):os.makedirs(save_dir)
    search_dir = PREDICT_RESULT_IMG
    image_path_list = glob.iglob(os.path.join(search_dir,'**','*.png'))
    
    pbar = tqdm.tqdm(image_path_list)
    PREFIX = colorstr("Search FP images")
    pbar.desc = f'{PREFIX}'
    for img_path in pbar:
        #print(img_path)
        GT, predict = Analysis_img_path(img_path)
        #print(GT," ",predict)
        if not predict == GT:
            save_label_dir = os.path.join(save_dir,GT)
            if not os.path.exists(save_label_dir):os.makedirs(save_label_dir)
            shutil.copy(img_path,save_label_dir)
            