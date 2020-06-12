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


def get_unit_vector(start,end):
   return (end-start)/np.linalg.norm(end-start)

def get_feature (Point1,Point2): # return separated point of vector
    X1=Point1[0]
    Y1=Point1[1]
    X2=Point2[0]
    Y2=Point2[1]

    #create the points 
    #number_of_points=number_of_no_foot_feature
    number_of_points=4
    xs=np.linspace(X1,X2,number_of_points+2)
    ys=np.linspace(Y1,Y2,number_of_points+2)

    #print them
    #for i in range(len(xs)):
    #    print (xs[i],ys[i])

    #reshape
    X= np.array([[xs],[ys]])
    X=X.transpose()
    X=np.squeeze(X) #example 12,1,2 -> # 12,2
    return X


def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros

def get_dist(start,end):
   return np.linalg.norm(end-start)

def get_feature (Point1,Point2): # return separated point of vector
    X1=Point1[0]
    Y1=Point1[1]
    X2=Point2[0]
    Y2=Point2[1]

    #create the points 
    #number_of_points=number_of_no_foot_feature
    number_of_points=3
    xs=np.linspace(X1,X2,number_of_points+2)
    ys=np.linspace(Y1,Y2,number_of_points+2)

    #print them
    #for i in range(len(xs)):
    #    print (xs[i],ys[i])

    #reshape
    X= np.array([[xs],[ys]])
    X=X.transpose()
    X=np.squeeze(X) #example 12,1,2 -> # 12,2
    return X

def in_feature(keypoints):
  x0,y0,z0=[],[],[]
  X=[]
  #print('print keypoints',keypoints) 
  min_dist=[]
  count=0
  all_person_count=len(keypoints[0])
  frame_count=len(keypoints)
  print('TEST',all_person_count)
  frame_t=[]
  for i in range(all_person_count):
    p1=[]
    for j in range(frame_count):
      print('print frame ','[',j,']','person','[',i,']',i,keypoints[j][i][1][0])
      a=keypoints[j][i]
      a=test_extract_featrue(keypoints[j][i])
      #print('type = a',type(a),np.array(a).shape)
      a=np.array(a)
      #print ('Evan print extracted feature',a)
      if ( len(p1)==0 ): #(1,25,3)  first person
        p1=np.expand_dims(a,axis=0)
      else:
        p1=np.vstack((p1,np.expand_dims(a,axis=0)))
      #print ('p1.shape',p1.shape,p1) 
 
    if ( len(frame_t)==0 ): #(1,25,3)  first person
      frame_t=np.expand_dims(p1,axis=0)
    else:
      frame_t=np.vstack((frame_t,np.expand_dims(p1,axis=0))) 
    print('framt_t shape',frame_t.shape)  
  #print('verify',frame_t[0][0][1][0])
  #print('verify',frame_t[4][1][1],frame_t[4][3][1][0])

  return frame_t
 
    #center = np.array((float(960),float(540)))
    #min_dist.append(get_dist(nose,center))
  #center_person_index=min_dist.index(min(min_dist))
#  center_person_index=0
  #a=keypoints[center_person_index]
  #return X

def test_extract_featrue(a):
    x0,y0,z0=[],[],[]
    X=[]
    print('a[0][0]',a[0][0])
    Nose_x=a[0][0];Nose_y=a[0][1];Neck_x=a[1][0];Neck_y=a[1][1];RShoulder_x=a[2][0];
    RShoulder_y=a[2][1];
    LShoulder_x=a[5][0];LShoulder_y=a[5][1];MidHip_x=a[8][0];MidHip_y=a[8][1];RHip_x=a[9][0];
    RHip_y=a[9][1];RKnee_x=a[10][0];RKnee_y=a[10][1];RAnkle_x=a[11][0];RAnkle_y=a[11][1];LHip_x=a[12][0];LHip_y=a[12][1]
    LKnee_x=a[13][0];LKnee_y=a[13][1];LAnkle_x=a[14][0];LAnkle_y=a[14][1];LBigToe_x=a[19][0];LBigToe_y=a[19][1]
    LSmallToe_x=a[20][0];LSmallToe_y=a[20][1];LHeel_x=a[21][0];LHeel_y=a[21][1];RBigToe_x=a[22][0];RBigToe_y=a[22][1]
    RSmallToe_x=a[23][0];RSmallToe_y=a[23][1];RHeel_x=a[24][0];RHeel_y=a[24][1]
  #17 points
    nose= np.array((float(Nose_x),float(Nose_y)))
    neck= np.array((float(Neck_x),float(Neck_y)))

    #0611 if shoulder is difference side.
    if (RShoulder_x==0):
      RShoulder_x=LShoulder_x
    if (RShoulder_y==0):
      RShoulder_y=LShoulder_y
    if (LShoulder_x==0):
      LShoulder_x=RShoulder_x
    if (LShoulder_y==0):
      LShoulder_y=RShoulder_y

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

    #Evan 0314 start 
    top=get_feature(rshoulder,lshoulder)    # return separated point of vector
    bottom=get_feature(rhip,lhip) 
    left=get_feature(lshoulder,lhip)
    right=get_feature(rshoulder,rhip)
    lagR=get_feature(rhip,rknee)
    lagL=get_feature(lhip,lknee)   
    #remove foot start
#            lag_toeR=get_feature(rhip,rbigtoe)
#            lag_toeL=get_feature(lhip,lbigtoe)
#            lag_toeRS=get_feature(rhip,rsmalltoe)
#            lag_toeLS=get_feature(lhip,lsmalltoe)
#            lag_heelL=get_feature(lhip,lheel)
#            lag_heelR=get_feature(rhip,rheel)
            #remove foot end

    #get_feature-> saperated a line into X parts
    test_feature=get_feature(lknee,rknee)
    all_feature=np.concatenate((top,bottom),axis=0)
    all_feature=np.concatenate((all_feature,left),axis=0)
    all_feature=np.concatenate((all_feature,right),axis=0)
    all_feature=np.concatenate((all_feature,lagR),axis=0)
    all_feature=np.concatenate((all_feature,lagL),axis=0)
    all_feature=np.concatenate((all_feature,test_feature),axis=0)
    #remove foot start
    #all_feature=np.concatenate((all_feature,lag_toeR),axis=0)
    #all_feature=np.concatenate((all_feature,lag_toeL),axis=0)
    #all_feature=np.concatenate((all_feature,lag_toeRS),axis=0)
    #all_feature=np.concatenate((all_feature,lag_toeLS),axis=0)
    #all_feature=np.concatenate((all_feature,lag_heelR),axis=0)
    #all_feature=np.concatenate((all_feature,lag_heelL),axis=0)
    #remove foot end
    #test_feature=np.concatenate((all_feature,test_feature),axis=0)

    #0405 modified unit vector
    top_uni=get_unit_vector(rshoulder,lshoulder)    # return separated point of vector
    bottom_uni=get_unit_vector(rhip,lhip)
    left_uni=get_unit_vector(lshoulder,rhip)
    right_uni=get_unit_vector(rshoulder,lhip)
    lagR_uni=get_unit_vector(rhip,lknee)
    lagL_uni=get_unit_vector(lhip,rknee)
    all_unit_vector=np.concatenate((top_uni,bottom_uni),axis=0)
    all_unit_vector=np.concatenate((all_unit_vector,left_uni),axis=0)
    all_unit_vector=np.concatenate((all_unit_vector,right_uni),axis=0)
    all_unit_vector=np.concatenate((all_unit_vector,lagR_uni),axis=0)
    all_unit_vector=np.concatenate((all_unit_vector,lagL_uni),axis=0)

    i=0
    f_people=[]
    for data in all_feature:
      x_uni=get_unit_vector(gravityXY,data)
      all_unit_vector=np.concatenate((all_unit_vector,x_uni),axis=0)
      i=i+1

    for unit_data in all_unit_vector:
      f_people.append(unit_data)
    #Evan 0314 end
    #if (lshoulder[0]>rshoulder[0]):
    x0.append(f_people) #Evan 0314 change feature for no foot
    return x0



