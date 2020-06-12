# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse

from evan_NN_nofoot_feature_backup import in_feature

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

#0418
sys.path.append('/home/evan/mp4_to_png/CONVERT_VIDEO_PIC/TESTER/ALL_TEST_DATA')

from gen_statistic import get_TESTER_data
import gen_statistic

xS,yS=gen_statistic.get_TESTER_data() # x,y for statistic.
CLASSES=7

#use cuda 
assert torch.cuda.is_available()
device = torch.device("cuda")

def draw_CLASS_statistic(class_rate):
  plt.ion()
  plt.show()
  class_plt=[]
  class_plt = list(range(1,7))
  class_plt.append(7)
  class_plt

  plt.plot(class_plt,class_rate,label="1M->9M",color="red",linewidth=2)
  plt.xlabel("Class")
  plt.ylabel("Accuracy")
  plt.ylim(0,1)
  plt.legend()
  plt.grid(True)
  plt.show()
  plt.savefig("CLASS_ACC_7P_1M9M_0425.png")
  plt.cla()


def draw_METER_statistic(Meter_rate):
  plt.ion()
  plt.show()

  meter_plt=[]
  meter_plt = list(range(1,9))
  meter_plt.append(9)
  meter_plt

  plt.plot(meter_plt,Meter_rate,label="CLASS1->CLASS7",color="red",linewidth=2)
  plt.xlabel("Meter")
  plt.ylabel("Accuracy")
  plt.ylim(0,1)
  plt.legend()
  plt.grid(True)
  plt.show()
  plt.savefig("MEMTER_ACC_7P_1M9M_0425.png")



def print_class_rate(in_data,all_people):
    all_data=in_data
    all_test_count=all_people
    class_rate = []  #1M->9M 
    for i in range(CLASSES): #class = 7
      class_rate.append(0)
    for i in range(CLASSES):
      class_rate[i]=all_data[:,i].sum()/all_test_count
    print('class_rate',class_rate)
    class_rate=np.array(class_rate)
    print('Total ACC Rate',class_rate.sum()/class_rate.size)
    return class_rate
def print_meter_rate(in_data):
    all_data=in_data
    tmp=[]
    Meter_rate = []  #1M->9M 
    #for i in range(9):
    #  Meter_rate.append(0)
    for i in range (9): #9M
      Px=0
      tmp=[]
      for j in range(i,63,9):
        #print('i=Meter',i,'j=index',j,'person num=',Px)
        Px=Px+1
        tmp.append(list(all_data[j]))
      tmp1=np.array(tmp)
      print('Total person Px=',Px,'in Meter=',i+1)
      print('class1',tmp1[0:,0].sum()/Px)  #class 1
      print('class2',tmp1[0:,1].sum()/Px)  #class 2
      print('class3',tmp1[0:,2].sum()/Px)  #class 3
      print('class4',tmp1[0:,3].sum()/Px)  #class 4
      print('class5',tmp1[0:,4].sum()/Px)  #class 5
      print('class6',tmp1[0:,5].sum()/Px)  #class 6
      print('class7',tmp1[0:,6].sum()/Px)  #class 7
      print('Meter Rate',i+1,tmp1.sum()/tmp1.size)
      Meter_rate.append(tmp1.sum()/tmp1.size)
    return Meter_rate
def draw_prediction(start,end,pred):
    fontsize=2
    pred_result=pred
    #pred_result=pred_result+5
    x1,y1,x2,y2=start[0],start[1],end[0],end[1]
    #print ('start',start,'end',end,"PRED",pred_result)
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
net =torch.load('/home/evan/mp4_to_png/0314/0406_1_9M_epoch80.pkl')
net =torch.load('/home/evan/mp4_to_png/0314/0422_1_9M.pkl')

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
y=[]
result_row=[]
result_col=[]
pred=0
meter=1
try:
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
   
    for index,person in enumerate(xS):
#        print ('person',person)    
        frame=cv2.imread(person)
        imageToProcess = frame
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum]) 
    
        x0=in_feature(datum.poseKeypoints) #return to draw in the frame
        x.append(x0)
        y.append(yS[index])
        if(len(x) == 7):
          if(meter==10):
            meter=1
          print('meter=',meter,'person=', person[83:85])
          meter=meter+1
          print('x==7 erase')
          print('y gt=',y)
          tensor = torch.ones((2,), dtype=torch.float32)
          X=tensor.new_tensor(x)
          X=X.unsqueeze(1) # the data formate should be [batch_size,1,30]
          X=X.cuda()
          out=net(X)
          _, pred_label = torch.max(out.data, 1)
          a=torch.tensor(np.array([0, 0, 0,0,0,0,0]))
          correct=0
          for i, value in enumerate(pred_label):
            a[int(value.item())]=a[int(value.item())]+1
            if(y[i]==value.item()):
              correct=correct+1
              result_col.append(1)
            else:
              result_col.append(0)
          result_row.append(result_col)            
          print ('correct=',correct)
          #print('predition label=',np.argmax(a))
          print('predition label=',pred_label)
          pred=np.argmax(a)
          result_col=[]
          x=[]
          y=[]

        noseX,noseY=datum.poseKeypoints[0][0][0],datum.poseKeypoints[0][0][1]
        x1,y1,newx,newy=900,400,1000,500
        start= np.array((float(noseX),float(noseY)))
        end=np.array((float(noseX)+100,float(noseY)))
        frame=datum.cvOutputData
        x1,y1,newxL,newyL,newxR,newyR=draw_prediction(start,end,pred)
        pt1=(int(x1), int(y1)-150)
        pt2=(int(newxR),int(newyR)-150)
        pt3=(int(newxL),int(newyL)-150)
        triangle_cnt = np.array( [pt1, pt2, pt3] )
        cv2.drawContours(frame, [triangle_cnt], 0, (0,255,0), -1)
        cv2.arrowedLine(frame,(int(x1), int(y1)-150),(int(newxL),int(newyL)-150),(0,0,255),2,tipLength = 0.2)
        cv2.arrowedLine(frame,(int(x1), int(y1)-150),(int(newxR),int(newyR)-150),(0,0,255),2,tipLength = 0.2)
    #DRAW picture
    
    all_test_count=len(result_row)
    all_data=np.array(result_row) 
    class_rate=print_class_rate(all_data,all_test_count)
    Meter_rate=print_meter_rate(all_data)

    draw_CLASS_statistic(class_rate)
    draw_METER_statistic(Meter_rate)
except Exception as e:
    # print(e)
    sys.exit(-1)
 
