# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse

from evan_NN_nofoot_feature_UNIT_VECTOR import in_feature

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

from shutil import copyfile
import re 

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
  plt.savefig("CLASS_ACC_7P_1M9M_unitV_0427.png")
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
  plt.savefig("MEMTER_ACC_7P_1M9M_unitV_0427.png")



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


dst='/home/evan/mp4_to_png/CONVERT_VIDEO_PIC/TESTER/ALL_PIC/'

for index,person in enumerate(xS):
  print ('person',person)    
  dot=person.find('.')
  slash=[m.start() for m in re.finditer('/', person)]
  start=slash[len(slash)-1]
  new_dst=dst+person[start+1:dot]+person[dot:]
  copyfile(person, new_dst)    
 
