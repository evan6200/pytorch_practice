import cv2
import numpy as np
import sys

print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv))
print ('usage:')
print ('0.preview_mp4.py video.avi png_folder/ label start end')
video_workspace='/home/evan/mp4_to_png/'
png_data_folder=sys.argv[2] + '/'
label=sys.argv[3]
count=0
prefix_name="pic"
prefix_name=label+"_"+prefix_name
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(sys.argv[1])
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

minute=13
start=minute*3600 
count=start # start from frame 0
start=int(sys.argv[4])
start=start*60
count=start
end=start+3600
end=int(sys.argv[5])
end=end*60
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
    if png_data_folder != "" :
        count=count+1
        count_str='%05d'% count
        filename=video_workspace+png_data_folder+count_str+"_"+label+".png"
        print (filename)
        if count >= start :
       # if count % 2 == 0 and count >= start :
            cv2.imwrite(filename,frame) 
    #        print filename 
        if count > end :
            break;
    #print filename
    # Press Q on keyboard to  exit
#    if cv2.waitKey(25) & 0xFF == ord('q'):
#      break
 
  # Break the loop

print ("in release") 
# When everything done, release the video capture object
cap.release()
print ("done" )
# Closes all the frames
cv2.destroyAllWindows()

