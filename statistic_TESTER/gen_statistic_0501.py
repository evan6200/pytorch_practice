import os
import sys

def fib(n):    # write Fibonacci series up to n
    a, b = 0, 1
    while a < n:
        print(a, end=' ')
        a, b = b, a+b
    print()
    return a

def test_print():
  print('test call this function get_TESTER_data')
  #path='/home/evan/mp4_to_png/CONVERT_VIDEO_PIC/TESTER/ALL_TEST_DATA/'
  path='/home/evan/mp4_to_png/CONVERT_VIDEO_PIC/TESTER_0430/ALL_TEST_DATA_0430/'
  root_dir=[name for name in os.listdir(path)]
  root_dir.sort()
  x=[]
  y=[]
  #for index_sub1,sub1_dir in enumerate(root_dir):
  #  print (sub1_dir)
  return root_dir


def get_TESTER_data():
  print('test call this function get_TESTER_data')
  #path='/home/evan/mp4_to_png/CONVERT_VIDEO_PIC/TESTER/ALL_TEST_DATA/'
  path='/home/evan/mp4_to_png/CONVERT_VIDEO_PIC/TESTER_0430/ALL_TEST_DATA_0430/'
  root_dir=[name for name in os.listdir(path)]

  #root_dir.sort()
  root_dir.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
  x=[]
  y=[]
  for index_sub1,sub1_dir in enumerate(root_dir):
    sub1_dir=path+sub1_dir
    if (os.path.isdir(sub1_dir) and sub1_dir != (path+'__pycache__')):
      sub1_dir_name=[name for name in os.listdir(sub1_dir)]
      sub1_dir_name.sort()
      for index_sub2,sub2_dir in enumerate(sub1_dir_name):
        #print(sub2_dir,'meter=',index_sub2)
        sub2_dir=sub1_dir+'/'+sub2_dir
        sub2_dir_name=[name for name in os.listdir(sub2_dir)]
        sub2_dir_name.sort()
        for index_sub3,sub3_file in enumerate(sub2_dir_name):
          image_path=sub2_dir+'/'+sub3_file
          #print('image_path=',image_path,'ground true=',index_sub3)
          x.append(image_path)
          y.append(index_sub3) #ground true
  return x,y
