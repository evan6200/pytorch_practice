import cv2
import numpy as np
import sys

print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv))
print ('usage:')
print ('0.preview_mp4.py video.avi png_folder/ label')

cap = cv2.VideoCapture(sys.argv[1])

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    # Display the resulting frame
    cv2.imshow('Frame',frame)
    # Press Q on keyboard to  exit
    k = cv2.waitKey(33)#ESC
    if k==27:    # Esc key to stop
        break
  else:
    break
 
  # Break the loop

print ( "in release" )
# When everything done, release the video capture object
cap.release()
print ("done" )
# Closes all the frames
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)
