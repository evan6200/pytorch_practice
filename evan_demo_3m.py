# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse

from evan_NN import net_model

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
#use cuda 
assert torch.cuda.is_available()
device = torch.device("cuda")

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
    if pred_result==0:
        angle=np.arctan2(1,0) #down
        newx = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newy = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1
    if pred_result==1:
        angle=np.arctan2(2.414,-1)
        newx = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newy = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1
    if pred_result==2:
        angle=np.arctan2(1,-1)
        newx = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newy = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1
    if pred_result==3:
        angle=np.arctan2(0.414,-1)
        newx = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newy = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1
    if pred_result==4:
        angle=np.arctan2(1,0.414)
        newx = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newy = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1
    if pred_result==5:
        angle=np.arctan2(1,1)
        newx = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newy = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1
    if pred_result==6:
        angle=np.arctan2(0.414,1)
        newx = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newy = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1
    if pred_result==7:
        angle=np.arctan2(1,1)
        newx = ((x2-x1)*np.cos(angle)-(y2-y1)*np.sin(angle)) + x1
        newy = ((x2-x1)*np.sin(angle)+(y2-y1)*np.cos(angle)) + y1
        cv2.arrowedLine(img2,(int(x1), int(y1)),(int(newx),int(newy)),(255,0,255),3)
    return x1,y1,newx,newy


class SoftMax(nn.Module):
    def __init__(self, initial_num_channels, num_classes, num_channels):
        super(SoftMax, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=initial_num_channels,
                      out_channels=256, kernel_size=3,stride=2),            
            nn.ReLU(inplace=True),
#            torch.nn.BatchNorm1d(128),
            nn.Conv1d(in_channels=256, out_channels=128,
                      kernel_size=3,stride=2),
            nn.ReLU(inplace=True),
            torch.nn.Dropout(0.2),
            nn.Conv1d(in_channels=128, out_channels=64,
                      kernel_size=3,stride=2),
            nn.ReLU(inplace=True),
#            torch.nn.BatchNorm1d(64,affine=True),
            torch.nn.Dropout(0.2),
            nn.Conv1d(in_channels=64,
                      out_channels=32, kernel_size=1,stride=1),

            nn.ReLU(inplace=True)

        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
#        print('x',x.size())       
        features = self.convnet(x).squeeze(dim=2)
#        print('features',features.size())
        prediction_vector = self.fc(features)
#        print('prediction_vector',prediction_vector.size())

        return prediction_vector

net =torch.load('/home/evan/mp4_to_png/MODEL_1213_9m_P1_P2.pkl')
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                           mode='min', factor=0.5,
                                           patience=1)
#class_weights = 1.0 / torch.tensor(class_label, dtype=torch.float32)
#class_weights=class_weights.to(device)
#criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion=nn.MSELoss()

print ('net',net)


# Import Openpose (Windows/Ubuntu/OSX)
__file__='evan_demo.py'
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
pred=0
try:
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    cap = cv2.VideoCapture('/home/evan/mp4_to_png/3M_test.avi')
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
      print("Error opening video stream or file")
    
    while(cap.isOpened()):
      # Capture frame-by-frame
      ret, frame = cap.read()
      if ret == True:
        imageToProcess = frame
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])
        # Display the resulting frame
        #cv2.imshow('Frame',datum.cvOutputData)
        x0=net_model(datum.poseKeypoints) #return to draw in the frame
        x.append(x0)
        if(len(x) == 10):
          print('x==5 erase')
          tensor = torch.ones((2,), dtype=torch.float32)
          X=tensor.new_tensor(x)
          total_object_count=list(X.size())[0]
          feature_size=list(X.size())[1]
          x1=torch.zeros(total_object_count,feature_size,feature_size)
          for i,data in enumerate(X):
            x1[i]=X[i]
          X=x1
          #print ('X size',X.size(), 'DATA',X)
          X=X.cuda()
          out=net(X)

          _, pred_label = torch.max(out.data, 1)
          a=torch.tensor(np.array([0, 0, 0,0,0,0,0]))

          for value in pred_label:
            a[int(value.item())]=a[int(value.item())]+1
          print('predition label=',np.argmax(a))
          pred=np.argmax(a)
          x=[]

#        cv2.imshow('Frame',datum.cvOutputData)
        

        noseX,noseY=datum.poseKeypoints[0][0][0],datum.poseKeypoints[0][0][1]
#        print('DRAW predition label=',pred)
#        print('noseX',noseX,'noseY',noseY)
        x1,y1,newx,newy=900,400,1000,500
#        if(noseX > 0 and noxeY > 0):
        start= np.array((float(noseX),float(noseY)))
        end=np.array((float(noseX)+100,float(noseY)))
        frame=datum.cvOutputData
        x1,y1,newx,newy=draw_prediction(start,end,pred)
        #cv2.arrowedLine(frame,(int(x1), int(y1)),(int(newx),int(newy)),(255,0,255),3)
        cv2.arrowedLine(frame,(int(x1), int(y1)-150),(int(newx),int(newy)-150),(255,0,255),3)
        cv2.imshow('Frame',frame) 
        #print (datum.poseKeypoints)
        #break
      # Press Q on keyboard to  exit
      k = cv2.waitKey(33)#ESC
      if k==27:    # Esc key to stop
        break
#    imageToProcess = cv2.imread(args[0].image_path)
#    datum.cvInputData = imageToProcess
#    opWrapper.emplaceAndPop([datum])

    # Display Image
#    print("Body keypoints: \n" + str(datum.poseKeypoints))
#    cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)
    #cv2.waitKey(0)
    #cap.release()
except Exception as e:
    # print(e)
    sys.exit(-1)
