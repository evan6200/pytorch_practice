import os
import sys


def get_TESTER_data():
  path='/home/evan/mp4_to_png/CONVERT_VIDEO_PIC/TESTER/ALL_TEST_DATA/'
  root_dir=[name for name in os.listdir(path)]

  root_dir.sort()
  x=[]
  y=[]
  for index_sub1,sub1_dir in enumerate(root_dir):
    if (os.path.isdir(sub1_dir) and sub1_dir !='__pycache__'):
      sub1_dir_name=[name for name in os.listdir(sub1_dir)]
      sub1_dir_name.sort()
      for index_sub2,sub2_dir in enumerate(sub1_dir_name):
        #print(sub2_dir,'meter=',index_sub2)
        sub2_dir=sub1_dir+'/'+sub2_dir
        sub2_dir_name=[name for name in os.listdir(sub2_dir)]
        sub2_dir_name.sort()
        for index_sub3,sub3_file in enumerate(sub2_dir_name):
          image_path=path+'/'+sub2_dir+'/'+sub3_file
          print('image_path=',image_path,'ground true=',index_sub3)
          x.append(image_path)
          y.append(index_sub3)
  return x,y
