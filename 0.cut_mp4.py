import cv2
import numpy as np
import sys

print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv))
print ('usage:')
print ('0.preview_mp4.py video.avi png_folder/ label')

#video_workspace='/home/evan/mp4_to_png/'
#png_data_folder=sys.argv[2] + '/'
#label=sys.argv[3]
#count=0
#prefix_name="pic"
#prefix_name=label+"_"+prefix_name
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
video_path=sys.argv[1]
video_path='/home/evan/gopro_pic/2019.11.31/GOPR0102.MP4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter('output_1.avi', fourcc, 30, (1920,1080))
 
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
#    cv2.imshow('Frame',frame)
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

