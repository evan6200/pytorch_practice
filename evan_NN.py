import numpy as np
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

def get_dist(start,end):
   return np.linalg.norm(end-start)

def net_model(keypoints):
  x0,y0,z0=[],[],[]   
  #print('print keypoints',keypoints) 
  min_dist=[]
  for a in keypoints:
    noseX,noseY=a[0][0],a[0][1]
    nose= np.array((float(noseX),float(noseY)))
    center = np.array((float(960),float(540)))
    min_dist.append(get_dist(nose,center))
  center_person_index=min_dist.index(min(min_dist))
#  center_person_index=0
  a=keypoints[center_person_index]
  Nose_x=a[0][0];Nose_y=a[0][1];Neck_x=a[1][0];Neck_y=a[1][1];RShoulder_x=a[2][0];
  RShoulder_y=a[2][1];
  LShoulder_x=a[5][0];LShoulder_y=a[5][1];MidHip_x=a[8][0];MidHip_y=a[8][1];RHip_x=a[9][0];
  RHip_y=a[9][1];RKnee_x=a[10][0];RKnee_y=a[10][1];RAnkle_x=a[11][0];RAnkle_y=a[11][1];LHip_x=a[12][0];LHip_y=a[12][1]
  LKnee_x=a[13][0];LKnee_y=a[13][1];LAnkle_x=a[14][0];LAnkle_y=a[14][1];LBigToe_x=a[19][0];LBigToe_y=a[19][1]
  LSmallToe_x=a[20][0];LSmallToe_y=a[20][1];LHeel_x=a[21][0];LHeel_y=a[21][1];RBigToe_x=a[22][0];RBigToe_y=a[22][1]
  RSmallToe_x=a[23][0];RSmallToe_y=a[23][1];RHeel_x=a[24][0];RHeel_y=a[24][1]
   
  #print("Evan_PRINT",Nose_x,Nose_y)
  
  #17 points
  nose= np.array((float(Nose_x),float(Nose_y))) 
  neck= np.array((float(Neck_x),float(Neck_y)))
  rshoulder = np.array((float(RShoulder_x),float(RShoulder_y)))
  lshoulder = np.array((float(LShoulder_x),float(LShoulder_y))) 
  mhip = np.array((float(MidHip_x),float(MidHip_y)))
  rhip = np.array((float(RHip_x),float(RHip_y)))
  rknee = np.array((float(RKnee_x),float(RKnee_y)))
  rankle = np.array((float(RAnkle_x),float(RAnkle_y)))
  lhip = np.array((float(LHip_x),float(LHip_y)))
  lknee = np.array((float(LKnee_x),float(LKnee_y)))
  lankle = np.array((float(LAnkle_x),float(LAnkle_y)))
  lbigtoe =np.array((float(LBigToe_x),float(LBigToe_y)))
  lsmalltoe = np.array((float(LSmallToe_x),float(LSmallToe_y)))
  lheel = np.array((float(LHeel_x),float(LHeel_y)))
  rbigtoe = np.array((float(RBigToe_x),float(RBigToe_y)))
  rsmalltoe = np.array((float(RSmallToe_x),float(RSmallToe_y)))
  rheel =  np.array((float(RHeel_x),float(RHeel_y))) 
            
  #caculate area of body 
  a=np.linalg.norm(lshoulder-rshoulder)
  b=np.linalg.norm(rshoulder-rhip)
  c=np.linalg.norm(lshoulder-rhip)
  #print('a,b,c',a,b,c)
  a1=np.linalg.norm(lshoulder-lhip)
  b1=np.linalg.norm(lhip-rhip)
  #print('a1,b1',a1,b1)
  #area 1
  s1=(a+b+c)/2
  area1=np.sqrt(s1*(s1-a)*(s1-b)*(s1-c))
  #area 2
  s2=(a1+b1+c)/2
  area2=np.sqrt(s2*(s2-a)*(s2-b)*(s2-c))
  area=area1+area2
  coorDelta2=[(lshoulder[0]+lhip[0]+rhip[0])/3,(lshoulder[1]+lhip[1]+rhip[1])/3]
  coorDelta1=[(lshoulder[0]+rshoulder[0]+rhip[0])/3,(lshoulder[1]+rshoulder[1]+rhip[1])/3]
  gravityXY=[(coorDelta1[0]*s1+coorDelta2[0]*s2)/(s1+s2),(coorDelta1[1]*s1+coorDelta2[1]*s2)/(s1+s2)]
  # print nose
  d_nose=get_dist(gravityXY,nose)/get_dist(neck,mhip) 
  d_neck=get_dist(gravityXY,neck)/get_dist(neck,mhip)
  d_rshoulder=get_dist(gravityXY,rshoulder)/get_dist(neck,mhip)
  d_lshoulder=get_dist(gravityXY,lshoulder)/get_dist(neck,mhip)
  d_mhip=get_dist(gravityXY,mhip)/get_dist(neck,mhip)
  d_rhip=get_dist(gravityXY,rhip)/get_dist(neck,mhip)
  d_rknee=get_dist(gravityXY,rknee)/get_dist(neck,mhip)
  d_rankle=get_dist(gravityXY,rankle)/get_dist(neck,mhip)
  d_lhip=get_dist(gravityXY,lhip)/get_dist(neck,mhip)
  d_lknee=get_dist(gravityXY,lknee)/get_dist(neck,mhip)
  d_lankle=get_dist(gravityXY,lankle)/get_dist(neck,mhip)
  d_lbigtoe=get_dist(gravityXY,lbigtoe)/get_dist(neck,mhip)
  d_lsmalltoe=get_dist(gravityXY,lsmalltoe)/get_dist(neck,mhip)
  d_lheel=get_dist(gravityXY,lheel)/get_dist(neck,mhip)
  d_rbigtoe=get_dist(gravityXY,rbigtoe)/get_dist(neck,mhip)
  d_rsmalltoe=get_dist(gravityXY,rsmalltoe)/get_dist(neck,mhip)
  d_rheel=get_dist(gravityXY,rheel)/get_dist(neck,mhip) 
  x0=[d_neck,d_rshoulder,d_lshoulder,d_mhip,d_rhip,d_rknee,d_rankle,d_lhip,d_lknee,d_lankle,d_lbigtoe,d_lsmalltoe,d_lheel,d_rbigtoe,d_rsmalltoe,d_rheel]
  
#  x=tensor.new_tensor(x0)
#  total_object_count=list(x.size())[0]
#  feature_size=list(x.size())[1]
#  x1=torch.zeros(total_object_count,feature_size,feature_size)
#  for i,data in enumerate(x):
#    x1[i]=x[i]
#  x=x1
  return x0
