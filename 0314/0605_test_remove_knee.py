import time
import sys
import cv2
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
from shutil import copyfile

number_of_no_foot_feature=4

PLT_UI=False

epoch=800
if (len(sys.argv)==2):
  epoch=sys.argv[1]

print('argv LEN',len(sys.argv))
print('EPOCH =',epoch)

#use cuda 
assert torch.cuda.is_available()
device = torch.device("cuda")

keypoint_video_6mP1_P2_3mP1="/home/evan/mp4_to_png/mix_3M_6M/3P1_6P1_P2/keypoint_mix/"
keypoint_video_9m_P1TEST="/home/evan/mp4_to_png/9m_train_data/save_video_p3/keypoint_all_P3TEST/" 
k_6mP1="/home/evan/mp4_to_png/6m_train_data/save_video_p1/keypoint_all/"
k_6mP2="/home/evan/mp4_to_png/6m_train_data/save_video_p2/keypoint_all_P2TEST/"
k_6mP3="/home/evan/mp4_to_png/6m_train_data/save_video_p3/keypoint_all_P3TEST/"

k_9mP1="/home/evan/mp4_to_png/9m_train_data/save_video_p1/keypoint_all_P1TEST/"
k_9mP2="/home/evan/mp4_to_png/9m_train_data/save_video_p2/keypoint_all_P2TEST/"
k_9mP3="/home/evan/mp4_to_png/9m_train_data/save_video_p3/keypoint_all_P3TEST/"

k_3mP1="/home/evan/mp4_to_png/3m/save_video_p1/keypoint_3MP1/"

k_MIX_3P1_6P1P2="/home/evan/mp4_to_png/mix_3M_6M/3P1_6P1_P2/keypoint_mix/"
k_MIX_3P1_6M_P1P2P3="/home/evan/mp4_to_png/mix_3M_6M/3P1_6P1_P2_P3/keypoint_mix/"
k_MIX_3P1_6M_P1P2_9P1="/home/evan/mp4_to_png/mix_3M_6M9M/3P1_6P1_P2_9P1/keypoint_mix/"

#0406
k_MIX_0406_1_9M="/home/evan/mp4_to_png/CONVERT_VIDEO_PIC/train_sample_0406/keypoint_0406_1M_9M/"

#0425
k_MIX_7PERSON="/home/evan/512_DISK/train_sample_7person/keypoints/7P_keypointsq_mod3/"

#0427
k_EVAN_1_9_WITH_CLOTHES='/home/evan/mp4_to_png/CONVERT_VIDEO_PIC/TESTER/ALL_PIC_KEYPOINT/'

#train C
k_MIX_7PERSON_C='/home/evan/512_DISK/train_sample_7person/keypoints/7P_keypointsq_mod3_C/'

k_MIX_7PERSON_EVAN='/home/evan/512_DISK/train_sample_7person/keypoints/7P_keypoints_mod3_EVAN/'

k_MIX_7PERSON_Evan3M6M='/home/evan/512_DISK/train_sample_7person/keypoints/7P_keypointsq_mod3_Evan_3M_6M/'

k_MIX_16PERSON='/home/evan/512_DISK/train_sample_7person/keypoints_P8_P16/16P_keypoints_mod3/'

#0501  #MOD20 -> 16Person
final_test='/home/evan/512_DISK/train_sample_7person/keypoints_P8_P16/16P_MOD20/'
final_box_test='/home/evan/512_DISK/train_sample_7person/keypoints_P8_P16/18P_MOD20_WITH_BOX/'

#0501
#training sample
training_keypoint=final_box_test
#test sample
#3M
#test_keypoint=k_6mP1
test_keypoint=k_EVAN_1_9_WITH_CLOTHES

#9M
#test_video="/home/evan/mp4_to_png/9m_train_data/save_video_p2/keypoint_all_P2TEST/"

#train=6m P1 P2 3m P1 = evan in home
#keypoint_video=keypoint_video_6mP1_P2_3mP1
#test=9m P1
#keypoint_video_P3TEST=keypoint_video_9m_P1TEST

files_TEST=[name for name in os.listdir(test_keypoint)]
files_TEST.sort()

files=[name for name in os.listdir(training_keypoint)]
files.sort()
#os.rename('a.txt', 'b.kml')
start=1
x0,y0,z0=[],[],[]
x0_test,y0_test,z0_test=[],[],[]
#index of count
label=7

def get_feature (Point1,Point2): # return separated point of vector
    X1=Point1[0]
    Y1=Point1[1]
    X2=Point2[0]
    Y2=Point2[1]

    #create the points 
    number_of_points=number_of_no_foot_feature
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

class_label=[]
for i in range(label):
    class_label.append(0)

def get_unit_vector(start,end):
   return (end-start)/np.linalg.norm(end-start)


def get_dist(start,end):
   return np.linalg.norm(end-start)

def read_data(data_files,key_point_path):
    x0,y0,z0=[],[],[]   
#assume only one people
    for index,filename in enumerate(data_files): #look up all keypoints
        index_label=filename.find('_')
        filepath=key_point_path+filename
        label=filename[6:7] # save label
#        print label
        if ( int(label[0]) == 0):
            class_label[0]=class_label[0]+1
        if ( int(label[0]) == 1):
            class_label[1]=class_label[1]+1
        if ( int(label[0]) == 2):
            class_label[2]=class_label[2]+1
        if ( int(label[0]) == 3):
            class_label[3]=class_label[3]+1
        if ( int(label[0]) == 4):
            class_label[4]=class_label[4]+1
        if ( int(label[0]) == 5):
            class_label[5]=class_label[5]+1
        if ( int(label[0]) == 6):
            class_label[6]=class_label[6]+1

#        print (filepath)
        with open(filepath,'r') as reader:  #load keypoint.json
            j_handler = json.loads(reader.read())
            person=np.zeros(len(j_handler['people'])) #how much person we have.
            #openpose_keypoints=np.zeros((len(j_handler['people'])) * 25*3) #25=> keypoints ,3=> (x,y,confidence)
            #openpose_keypoints=np.reshape(openpose_keypoints,(len(j_handler['people']),75))
            openpose_keypoints_one_person=np.zeros(1 * 25*3) 
            openpose_keypoints_one_person=np.reshape(openpose_keypoints_one_person,(1,75))
            
            min_dist=[]
            for k in j_handler['people']:
                noseX,noseY=k['pose_keypoints_2d'][0],k['pose_keypoints_2d'][1]
                nose= np.array((float(noseX),float(noseY)))
                center = np.array((float(960),float(540)))
                min_dist.append(get_dist(nose,center))
            center_person_index=min_dist.index(min(min_dist))
            
            openpose_keypoints_one_person=j_handler['people'][center_person_index]['pose_keypoints_2d'] #get 25 key points.
            a=openpose_keypoints_one_person
            Nose_x=a[0];Nose_y=a[1];Neck_x=a[3];Neck_y=a[4];RShoulder_x=a[6];RShoulder_y=a[7]
            LShoulder_x=a[15];LShoulder_y=a[16];MidHip_x=a[24];MidHip_y=a[25];RHip_x=a[27];RHip_y=a[28]
            RKnee_x=a[30];RKnee_y=a[31];RAnkle_x=a[33];RAnkle_y=a[34];LHip_x=a[36];LHip_y=a[37]
            LKnee_x=a[39];LKnee_y=a[40];LAnkle_x=a[42];LAnkle_y=a[43];LBigToe_x=a[57];LBigToe_y=a[58]
            LSmallToe_x=a[60];LSmallToe_y=a[61];LHeel_x=a[63];LHeel_y=a[64];RBigToe_x=a[66];RBigToe_y=a[67]
            RSmallToe_x=a[69];RSmallToe_y=a[70];RHeel_x=a[72];RHeel_y=a[73]

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
#            print nose
            #d_nose=get_dist(gravityXY,nose)/get_dist(neck,mhip) 
            #d_neck=get_dist(gravityXY,neck)/get_dist(neck,mhip)
            #d_rshoulder=get_dist(gravityXY,rshoulder)/get_dist(neck,mhip)
            #d_lshoulder=get_dist(gravityXY,lshoulder)/get_dist(neck,mhip)
            #d_mhip=get_dist(gravityXY,mhip)/get_dist(neck,mhip)
            #d_rhip=get_dist(gravityXY,rhip)/get_dist(neck,mhip)
            #d_rknee=get_dist(gravityXY,rknee)/get_dist(neck,mhip)
            #d_rankle=get_dist(gravityXY,rankle)/get_dist(neck,mhip)
            #d_lhip=get_dist(gravityXY,lhip)/get_dist(neck,mhip)
            #d_lknee=get_dist(gravityXY,lknee)/get_dist(neck,mhip)
            #d_lankle=get_dist(gravityXY,lankle)/get_dist(neck,mhip)
            #d_lbigtoe=get_dist(gravityXY,lbigtoe)/get_dist(neck,mhip)
            #d_lsmalltoe=get_dist(gravityXY,lsmalltoe)/get_dist(neck,mhip)
            #d_lheel=get_dist(gravityXY,lheel)/get_dist(neck,mhip)
            #d_rbigtoe=get_dist(gravityXY,rbigtoe)/get_dist(neck,mhip)
            #d_rsmalltoe=get_dist(gravityXY,rsmalltoe)/get_dist(neck,mhip)
            #d_rheel=get_dist(gravityXY,rheel)/get_dist(neck,mhip)

            #Evan 0314 start 
            top=get_feature(rshoulder,lshoulder)    # return separated point of vector
            bottom=get_feature(rhip,lhip) 
            left=get_feature(lshoulder,lhip)
            right=get_feature(rshoulder,rhip)
            left_2=get_feature(rshoulder,lhip)
            right_2=get_feature(lshoulder,rhip)

            lagR=get_feature(rhip,rknee)
            lagL=get_feature(lhip,lknee)   
            #remove foot start
            lag_toeR=get_feature(rhip,rbigtoe)
            lag_toeL=get_feature(lhip,lbigtoe)
            lag_toeRS=get_feature(rhip,rsmalltoe)
            lag_toeLS=get_feature(lhip,lsmalltoe)
            lag_heelL=get_feature(lhip,lheel)
            lag_heelR=get_feature(rhip,rheel)
            #remove foot end

            #get_feature-> saperated a line into X parts
            test_feature=get_feature(lknee,rknee)
            all_feature=np.concatenate((top,bottom),axis=0)
            all_feature=np.concatenate((all_feature,left),axis=0)
            all_feature=np.concatenate((all_feature,right),axis=0)
            all_feature=np.concatenate((all_feature,left_2),axis=0)
            all_feature=np.concatenate((all_feature,right_2),axis=0)

            all_feature=np.concatenate((all_feature,lagR),axis=0)
            all_feature=np.concatenate((all_feature,lagL),axis=0)
            all_feature=np.concatenate((all_feature,test_feature),axis=0)
            #remove foot start
            all_feature=np.concatenate((all_feature,lag_toeR),axis=0)
            all_feature=np.concatenate((all_feature,lag_toeL),axis=0)
            all_feature=np.concatenate((all_feature,lag_toeRS),axis=0)
            all_feature=np.concatenate((all_feature,lag_toeLS),axis=0)
            all_feature=np.concatenate((all_feature,lag_heelR),axis=0)
            all_feature=np.concatenate((all_feature,lag_heelL),axis=0)
            #remove foot end
            #test_feature=np.concatenate((all_feature,test_feature),axis=0)

            #0405 modified unit vector
            top_uni=get_unit_vector(rshoulder,lshoulder)    # return separated point of vector
            bottom_uni=get_unit_vector(rhip,lhip)
            left_uni=get_unit_vector(rshoulder,rhip)
            right_uni=get_unit_vector(lshoulder,lhip)
            left_uni_2=get_unit_vector(lshoulder,rhip)
            right_uni_2=get_unit_vector(rshoulder,lhip)

            lagR_uni=get_unit_vector(rhip,lknee)
            lagL_uni=get_unit_vector(lhip,rknee)
            all_unit_vector=np.concatenate((top_uni,bottom_uni),axis=0)
            all_unit_vector=np.concatenate((all_unit_vector,left_uni),axis=0)
            all_unit_vector=np.concatenate((all_unit_vector,right_uni),axis=0)
            all_unit_vector=np.concatenate((all_unit_vector,left_uni_2),axis=0)
            all_unit_vector=np.concatenate((all_unit_vector,right_uni_2),axis=0)

            all_unit_vector=np.concatenate((all_unit_vector,lagR_uni),axis=0)
            all_unit_vector=np.concatenate((all_unit_vector,lagL_uni),axis=0)


            i=0
            f_people=[]
            #print('all_feature.len=',len(all_feature))
            #print('all_feature=',all_feature)
            #print('all_unit_vector=',len(all_unit_vector))

            for data in all_feature:
                #x=get_dist(gravityXY,data)/get_dist(neck,mhip) 
                x_uni=get_unit_vector(gravityXY,data)
                all_unit_vector=np.concatenate((all_unit_vector,x_uni),axis=0)
                #print('x_unit',x_uni)
                i=i+1
                #f_people.append(x)
            for unit_data in all_unit_vector:
                #print('unit_data',unit_data)
                f_people.append(unit_data)
            #print('f_people',len(f_people))
            #print("Total feature=",i)
            #Evan 0314 end

            #0501 skip error frame
            if (lshoulder[0]>rshoulder[0]):
              x0.append(f_people) #Evan 0314 change feature for no foot
              #target=[0,0,0,0,0,0,0] #total CLASS,one hot 
              #target[int(label)]= target[int(label)]+1
              #y0.append(target)
              y0.append(int(label)) #no one hot.
    return x0,y0

class SoftMax_1D(nn.Module):
    def __init__(self, initial_num_channels, num_classes, num_channels):
        super(SoftMax_1D, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=initial_num_channels,
                      out_channels=512, kernel_size=4,stride=3),                      
            nn.ReLU(inplace=True),
            #torch.nn.BatchNorm1d(256),
            nn.Conv1d(in_channels=512, out_channels=256,
                      kernel_size=4,stride=3,padding=0),
            nn.ReLU(inplace=True),
            torch.nn.Dropout(0.2),
            nn.Conv1d(in_channels=256, out_channels=256,
                      kernel_size=4,stride=3),
            nn.ReLU(inplace=True),
            #torch.nn.BatchNorm1d(256,affine=True),
            torch.nn.Dropout(0.2),
            nn.Conv1d(in_channels=256,
                      out_channels=256, kernel_size=3,stride=2),  #0427 kernel_size 2->3
            nn.ReLU(inplace=True),
            #torch.nn.BatchNorm1d(64,affine=True),
            torch.nn.Dropout(0.2),
            nn.Conv1d(in_channels=256,
                      out_channels=64, kernel_size=3,stride=2,padding=1),
            
            nn.ReLU(inplace=True)
        
        )
        self.fc1 = nn.Linear(64, 64)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        print('x shape',x.shape)  
        #print('x',x)     
        print('self.convnet(x).shape',self.convnet(x).shape)
        print('self.convnet(x)',self.convnet(x))
        features = self.convnet(x).squeeze(dim=2)
#        print('features',features.size())
        prediction_vector = self.fc(features)
#        print('prediction_vector',prediction_vector.size())
        #return prediction_vector
        return F.log_softmax(prediction_vector,dim=1)

#0303 test 
net = SoftMax_1D(initial_num_channels=1,num_classes=7,num_channels=64)
#print ('net',net)
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                           mode='min', factor=0.5,
                                           patience=1)
class_weights = 1.0 / torch.tensor(class_label, dtype=torch.float32)
class_weights=class_weights.to(device)
criterion = nn.CrossEntropyLoss() #(weight=class_weights)
#criterion=nn.MSELoss()

net.train()

tensor = torch.ones((2,), dtype=torch.float32)
tensor2 = torch.ones((2,), dtype=torch.float32)

x0,y0= read_data(files,training_keypoint) #training sample
x0_test,y0_test=read_data(files_TEST,test_keypoint)

print('len(x0)=',len(x0))

x_orig=tensor.new_tensor(x0)
y_orig=tensor2.new_tensor(y0)

total_object_count=list(x_orig.size())[0]
feature_size=list(x_orig.size())[1]
y=y_orig.type(torch.LongTensor)

x1=torch.zeros(total_object_count,feature_size,feature_size)
for i,data in enumerate(x_orig):
   x1[i]=x_orig[i]
x=x1

X=x_orig.unsqueeze(1)

#setup test data
x_test_orig=tensor.new_tensor(x0_test)
y_test_orig=tensor2.new_tensor(y0_test)

total_object_count=list(x_test_orig.size())[0]
feature_size=list(x_test_orig.size())[1]
y_test=y_test_orig.type(torch.LongTensor)

x1=torch.zeros(total_object_count,feature_size,feature_size)
for i,data in enumerate(x_test_orig):
   x1[i]=x_test_orig[i]
x_test=x1

X_TEST=x_test_orig.unsqueeze(1)

#y=y.type(torch.LongTensor)
#y_test=y_test.type(torch.LongTensor)

BATCH_SIZE=1024
#training dataset

torch_dataset=Data.TensorDataset(x,y)
#test dataset
torch_dataset_test=Data.TensorDataset(x_test,y_test)

#print('Evan print y_test DATA',y_test)

#0303 test 
randomize = np.arange(len(X)) #randomize data
np.random.shuffle(randomize)
X = X[randomize] 
y = y[randomize]

#separated 30% for validate
validate_count=int(len(X)*0.2)
X_val = X[0:validate_count]
y_val = y[0:validate_count]

X = X[validate_count:]
y = y[validate_count:]

torch_dataset=Data.TensorDataset(X,y)
torch_dataset_test=Data.TensorDataset(X_TEST,y_test)
#30% for validation
torch_dataset_validate=Data.TensorDataset(X_val,y_val)

loader = Data.DataLoader(dataset=torch_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
# Validate data, 30% of training data
loader_test = Data.DataLoader(dataset=torch_dataset_validate,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)

#Out side data.
#loader_test = Data.DataLoader(dataset=torch_dataset_test,batch_size=7,shuffle=False,num_workers=2)

if (PLT_UI==True):
  plt.ion()
  plt.show()

epoch_plt=[]
loss_plt=[]
acc_train_plt=[]
acc_test_plt=[]
train_loss=0
test_loss=0
for epoch in range(int(epoch)):
    total_cnt=0
    correct_cnt, ave_loss = 0, 0
    ave_loss = 0
    i=0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        net.zero_grad()
        #target=batch_y.type(torch.LongTensor)
        i=i+1
        #print('Epoch:{}|num:{}|batch_x:{}|batch_y:{}'.format(epoch,i,batch_x,batch_y))
        #print('batch_x',batch_x.size())
        out=net(batch_x)
        loss = criterion(out, batch_y)
        ave_loss = ave_loss * 0.9 + loss.data * 0.1
        loss.backward()
        optimizer.step()
        _, pred_label = torch.max(out.data, 1)
        #_, real_label = torch.max(batch_y.data,1) # if cross entropy
        # correct_cnt += (pred_label == real_label).sum()
        #pred_label.eq(batch_y.view_as(pred_label)).sum().item() # if cross entropy
        correct_cnt += (pred_label == batch_y).sum() # if cross entropy
        total_cnt += batch_x.data.size()[0]

    loss_cpu=loss
    if (PLT_UI==True):
      epoch_plt.append(epoch)
      loss_plt.append(ave_loss)
      acc_train_plt.append(float(correct_cnt)/float(total_cnt))
      ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
      #plt.subplot(311) 
      ax1.plot(epoch_plt,loss_plt,color='tab:green')
      ax1.set_title('Train loss vs. epoches')
      #plt.subplot(312)
      ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=3)
      ax2.plot(epoch_plt,acc_train_plt,color='tab:blue')
      ax2.set_title('Train ACC vs. epoches')

      plt.pause(0.1)
#    print('--------------------------------------------------------')
#    print('TRAIN LOSS=',loss_cpu.item(),'Train total cnt=',total_cnt)
    train_loss=loss_cpu.item()
#    print('Train--loss=',loss,'ave_loss=',ave_loss,'correct_cnt=' ,correct_cnt,'total_cnt=', total_cnt)
    #print('Train-- acc in train =' ,float(correct_cnt)/float(total_cnt))
    train_acc=float(correct_cnt)/float(total_cnt)

    net.eval()
    correct_cnt, ave_loss = 0, 0
    ave_test_acc = 0
    total_cnt=0
    total_acc=0
    error_pic_number=[]
    error_predict=[]
    actual_label=[]
    all_label=[]
    all_error_label=[]
    i=0

    for t_batch_x, t_batch_y in loader_test:
        t_batch_x, t_batch_y = t_batch_x.cuda(),t_batch_y.cuda()
        i=i+1
        out=net(t_batch_x)
        loss = criterion(out, t_batch_y)
        _, pred_label = torch.max(out.data, 1)
        #_, real_label = torch.max(t_batch_y.data,1)
        total_cnt += t_batch_x.data.size()[0]
        #correct_cnt += (pred_label == real_label).sum()
        correct_cnt += (pred_label == t_batch_y).sum()
        ave_loss = ave_loss * 0.9 + loss.data * 0.1
        
        #ave_test_acc = ave_test_acc * 0.9 + (float(correct_cnt)/total_cnt) * 0.1
        ave_test_acc = (float(correct_cnt)/total_cnt)
        total_acc+=ave_test_acc
        #ave_test_acc = ave_test_acc/epoch
    #print('TEST LOSS=',ave_loss.item(),'TEST total_cnt',total_cnt)
    test_loss=ave_loss.item()
#    dataE=[]
#    for index,data in enumerate(error_pic_number):
        #print ('pic num={}, predict={}, label={}'.format(error_pic_number[index][0][0].item(),error_predict[index].item(),actual_label[index].item()))
#        dataE.append([error_pic_number[index][0][0].item(),error_predict[index].item(),actual_label[index].item()]);
#    dataE.sort()
#print('TESTING ==>>> epoch: {}, test loss: {:.6f}, acc: {:.3f},aveACC{:.3f}'.format(epoch, ave_loss, float(correct_cnt)/total_cnt,ave_test_acc))
    #print('all_label',len(all_label))
#    out_all_label=all_label
#    out_all_ERR=dataE
    #print('TESTING ==>>> epoch: {}, test loss: {:.6f}, acc: {:.3f}'.format(epoch, ave_loss, float(correct_cnt)/total_cnt))
    #print('TESTING ==>>> epoch: {}, test loss: {:.6f}, acc: {:.3f}, correct:{},total_cnt={}'.format(epoch, ave_loss, float(correct_cnt)/total_cnt, correct_cnt,total_cnt))
    #acc_test_plt.append(float(correct_cnt)/total_cnt)
    if (PLT_UI==True):
      acc_test_plt.append(ave_test_acc)
      ax3 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
      ax3.plot(epoch_plt,acc_test_plt,color='tab:orange')
      ax3.set_title('Test ACC vs. epoches,outside data 288 person')
    acc=float(correct_cnt)/total_cnt
    print ('train_acc',train_acc)
    print ('TEST ACC=',acc)
    #if acc>0.70 and ave_loss < 0.4 and train_acc > 0.75:
    if acc > 0.87 and  train_acc > 0.99:
        break

print('epoch=',epoch)
print('train_loss=',train_loss)
print('test_loss=',test_loss)
print ('train_acc',train_acc)
print ('TEST ACC=',acc)
date=time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
print('date=',date)
save_name="0605_demo_pkl/0605_uni_vector_"+ date + ".pkl"
torch.save(net,save_name)
dst="0605_demo_uni_vector_X_EPOCH.pkl"
copyfile(save_name,dst )
if (PLT_UI==True):
  plt.savefig("PIC_0605_DEMO_uni_vector.png")
