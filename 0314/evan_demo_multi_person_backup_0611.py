# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse

from evan_NN_nofoot_feature import in_feature

import torch


import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch.nn as nn
import torch.optim
import sklearn
from torch.autograd import Variable
from sklearn import datasets
import torch.utils.data as Data
import matplotlib.pyplot as plt

from threading import Thread 
import time

#if (len(sys.argv) <=1):
#  print('usage: python evan_demo_multi_person.py source.avi')
#  sys.exit()

def fire(data):
  print('in thread print', data )  


def timer(name,delay,times,d):
    print("計時器: "+ name + "開始" )
    while times > 0:
        time.sleep(delay)
        print(name + ": " + str(time.ctime(time.time())))
        times -= 1
        print(np.array(d).shape)
    print("計時器: " + name + "完成")

#use cuda 
assert torch.cuda.is_available()
device = torch.device("cuda")

def get_dist(start,end):
   return np.linalg.norm(end-start)

def draw_prediction(start,end,pred):
    fontsize=2
    pred_result=pred
    #pred_result=pred_result+5
    x1,y1,x2,y2=start[0],start[1],end[0],end[1]
    print ('start',start,'end',end,"PRED",pred_result)
    moved=(np.linalg.norm(end-start)/2)
    x2=x1+moved
    y1=y1+moved
    y2=y2+moved
    areaLR=np.pi/6
    if pred_result==0:
        angle=np.arctan2(1,0) #down
        newx = ((x2-x1)*np.cos(angle+np.pi/6)-(y2-y1)*np.sin(angle+np.pi/6)) + x1
        newy = ((x2-x1)*np.sin(angle+np.pi/6)+(y2-y1)*np.cos(angle+np.pi/6)) + y1
        angle=angle+areaLR
        newxL = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newyL = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1
        angle=angle-(2*areaLR)
        newxR = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newyR = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1
    if pred_result==1:
        angle=np.arctan2(2.414,-1)
        newx = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newy = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1
        angle=angle+areaLR
        newxL = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newyL = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1
        angle=angle-(2*areaLR)
        newxR = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newyR = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1

    if pred_result==2:
        angle=np.arctan2(1,-1)
        newx = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newy = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1
        angle=angle+areaLR
        newxL = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newyL = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1
        angle=angle-(2*areaLR)
        newxR = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newyR = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1

    if pred_result==3:
        angle=np.arctan2(0.414,-1)
        newx = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newy = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1
        angle=angle+areaLR
        newxL = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newyL = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1
        angle=angle-(2*areaLR)
        newxR = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newyR = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1

    if pred_result==4:
        angle=np.arctan2(1,0.414)
        newx = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newy = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1
        angle=angle+areaLR
        newxL = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newyL = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1
        angle=angle-(2*areaLR)
        newxR = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newyR = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1

    if pred_result==5:
        angle=np.arctan2(1,1)
        newx = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newy = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1
        angle=angle+areaLR
        newxL = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newyL = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1
        angle=angle-(2*areaLR)
        newxR = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newyR = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1

    if pred_result==6:
        angle=np.arctan2(0.414,1)
        newx = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newy = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1
        angle=angle+areaLR
        newxL = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newyL = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1
        angle=angle-(2*areaLR)
        newxR = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newyR = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1

    if pred_result==7:
        angle=np.arctan2(1,1)
        newx = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newy = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1
        angle=angle+areaLR
        newxL = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newyL = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1
        angle=angle-(2*areaLR)
        newxR = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newyR = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1

    return x1,y1,newxL,newyL,newxR,newyR


class SoftMax_1D(nn.Module):
    def __init__(self, initial_num_channels, num_classes, num_channels):
        super(SoftMax_1D, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=initial_num_channels,
                      out_channels=256, kernel_size=3,stride=2,dilation=2),                  
            nn.ReLU(inplace=True),
#            torch.nn.BatchNorm1d(128),
            nn.Conv1d(in_channels=256, out_channels=256,
                      kernel_size=3,stride=2,padding=0,dilation=2),
            nn.ReLU(inplace=True),
            torch.nn.Dropout(0.2),
            nn.Conv1d(in_channels=256, out_channels=256,
                      kernel_size=3,stride=2),
            nn.ReLU(inplace=True),
#            torch.nn.BatchNorm1d(64,affine=True),
            torch.nn.Dropout(0.2),
            nn.Conv1d(in_channels=256,
                      out_channels=64, kernel_size=2,stride=2),

            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Linear(64, 64)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
#        print('x',x.size())       
        features = self.convnet(x).squeeze(dim=2)
#        print('features',features.size())
        prediction_vector = self.fc(features)
#        print('prediction_vector',prediction_vector.size())

        return prediction_vector

#net =torch.load('/home/evan/mp4_to_png/0314/0316_1D_without_foot.pkl')
net =torch.load('0322_no_foot.pkl')
#net = torch.load('0605_demo_pkl/0605_uni_vector_2020-06-12_13:00:50.pkl')
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                           mode='min', factor=0.5,
                                           patience=1)
#class_weights = 1.0 / torch.tensor(class_label, dtype=torch.float32)
#class_weights=class_weights.to(device)
criterion = nn.CrossEntropyLoss()#(weight=class_weights)
#criterion=nn.MSELoss()


print ('net',net)


# Import Openpose (Windows/Ubuntu/OSX)
__file__='evan_demo_6m_0316_1DCNN.py '
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Change these variables to point to the correct folder (Release/x64 etc.) 
    sys.path.append('/home/evan/openpose/build/python')
    print ('import openpose')
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
    from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "/home/evan/openpose/models/"

# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1])-1: next_item = args[1][i+1]
    else: next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-','')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-','')
        if key not in params: params[key] = next_item

# Construct it from system arguments
# op.init_argv(args[1])
# oppython = op.OpenposePython()
x=[]
all_P_F=[]
pred=0
last_frame_people=0
fps=0
noseX,noseY=0,0
multi_pred=[]
collect_frame=0
try:
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    #cap = cv2.VideoCapture('/home/evan/mp4_to_png//6MP3_test.avi')
    #cap = cv2.VideoCapture('/home/evan/mp4_to_png/0314_4M_EVAN.avi')
    #cap = cv2.VideoCapture('/home/evan/512_DISK/0611_for_demo/multi_person/GOPR0354.MP4')
    cap = cv2.VideoCapture(sys.argv[1])
    #cap = cv2.VideoCapture('avi/3M_test.avi')
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
      print("Error opening video stream or file")
#    t1 = Thread(target=timer,args=("程式1",1,5,all_P_F))
#    t1.start() 
    while(cap.isOpened()):
      # Capture frame-by-frame
      ret, frame = cap.read()
      if ret == True:
        imageToProcess = frame
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])
       
        all_P_F.append(datum.poseKeypoints)
        
        if ( 5 == len(all_P_F)):
          collect_frame=len(all_P_F)
          fps=0
          x0=all_P_F[0]#t-1  , the first frame is t-1
          all_P_F.pop(0)
          x1=np.expand_dims(x0,axis=0) #(1,6,25,3) the first frame
          frame_t=[]
          for frame in all_P_F: #other frames is t t+1...t+N
            fps=fps+1
            d=[]
            p1=[]
            for p in x0:
              all_dist=[]
              start=np.array(p[1][0],p[1][1])
              for pT in frame: #T frame
                end=np.array(pT[1][0],pT[1][0])
                d=get_dist(start,end)
                all_dist.append(d)
              np_all_dist=np.array(all_dist)
              idx=np.argmin(np_all_dist) #find nearest people
              if(np_all_dist[idx] > 20): #it is to far, not belone with that guy.
                #print('dist bigger than 20',np_all_dist[idx])
                #print('frame shape',frame.shape,'x0.shape',x0.shape)
                zero_arr = (25,3) 
                zero_arr=np.zeros(zero_arr)
                person=zero_arr #if this person disaper , give zero
              else:
                person=frame[idx]
              if ( len(p1)==0 ): #(1,25,3)  first person
                p1=np.expand_dims(person,axis=0) 
              else:
                p1=np.vstack((p1,np.expand_dims(person,axis=0))) #(person++,feature,XY_C) (2,25,3)
            frame_t=np.expand_dims(p1,axis=0) #(1,6,25,3)
            np.vstack((x1,frame_t)).shape
            x1=np.vstack((x1,frame_t))  #(2,6,25,3) (frame++,person,feature,)
            #[Every Frame END]  
            #break 
            print('x1.shape=>',x1.shape)    
          
          #all_P_F=[]
          #break 
          x=in_feature(x1) #return to draw in the frame       
          print('finish feature extracted') 
          person=0
          frame=datum.cvOutputData
          for p in x:
            pX=torch.from_numpy(p).float()  
            pX=pX.unsqueeze(1)
            pX=pX.cuda()
            out=net(pX)
            _, pred_label = torch.max(out.data, 1)
            a=torch.tensor(np.array([0, 0, 0,0,0,0,0]))

            for value in pred_label:
              a[int(value.item())]=a[int(value.item())]+1
              print('predition label=',np.argmax(a))
              pred=np.argmax(a)
            #noseX,noseY=x0[person][1][0],x0[person][1][1]
            #print('neckXY',x0[person][1][0],x0[person][1][1],'pred',pred)
            multi_pred.append([x0[person][1][0],x0[person][1][1],pred])      
            person=person+1
          all_P_F=[]

      frame=datum.cvOutputData
      #draw multi person    
      #print('multi_pred',multi_pred)
      if (collect_frame and len(multi_pred) >0):
        for rander_person in multi_pred:
          noseX,noseY,draw_pred=rander_person[0],rander_person[1],rander_person[2]
          start= np.array((float(noseX),float(noseY)))
          end=np.array((float(noseX)+100,float(noseY)))
          x1,y1,newxL,newyL,newxR,newyR=draw_prediction(start,end,draw_pred)
          pt1=(int(x1), int(y1)-150)
          pt2=(int(newxR),int(newyR)-150)
          pt3=(int(newxL),int(newyL)-150)
          triangle_cnt = np.array( [pt1, pt2, pt3] )
          cv2.drawContours(frame, [triangle_cnt], 0, (0,255,0), -1)
          cv2.arrowedLine(frame,(int(x1), int(y1)-150),(int(newxL),int(newyL)-150),(0,0,255),2,tipLength = 0.2)
          cv2.arrowedLine(frame,(int(x1), int(y1)-150),(int(newxR),int(newyR)-150),(0,0,255),2,tipLength = 0.2)
        collect_frame=0
        multi_pred=[]
      cv2.imshow('Frame',frame)
      # Press Q on keyboard to  exit
      k = cv2.waitKey(33)#ESC
      if k==27:    # Esc key to stop
        break
    #cap.release()
except Exception as e:
    # print(e)
    sys.exit(-1)
