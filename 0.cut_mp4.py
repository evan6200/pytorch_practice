import cv2
import numpy as np
import sys


video_path="/home/evan/gopro_pic/2019.12.19/3M_training.MP4"
#video_path="/home/evan/gopro_pic/2019.11.31/15M.MP4"
cap = cv2.VideoCapture(video_path)
print ('test')
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter('TEST.avi', fourcc, 30, (1920,1080))
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

minute=12
start=minute*3600 
start=16200+2400 
count=start # start from frame 0
end=start+3600
end=24600
cap.set(1,count)
#print "set 14400"
#print cap.isOpened()
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    # Display the resulting frame
      cv2.imshow('Frame',frame)
#    if png_data_folder != "" :
      count=count+1
#        count_str='%05d'% count
#        filename=video_workspace+png_data_folder+count_str+".png"
      if count % 2 == 0 and count >= start :
          #cv2.imwrite(filename,frame) 
        out.write(frame)
      if count > end :
          break;
# When everything done, release the video capture object
cap.release()
print ("done" )
# Closes all the frames
cv2.destroyAllWindows()

