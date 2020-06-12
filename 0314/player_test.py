import cv2
import numpy as np
import sys

#translate MP4 to avi

if (len(sys.argv) <=3 ):
  print ('usage: python player_test.py src.mp4 output.avi start_second')
  sys.exit() 
cap = cv2.VideoCapture(sys.argv[1])

fourcc = cv2.VideoWriter_fourcc(*'XVID')

# $1 output file
# $2 input file
# $3 start_second
out = cv2.VideoWriter(sys.argv[2], fourcc, 30, (1920,1080))

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

cap.set(1,60*int(sys.argv[3])) #skip number of frame

while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    # Display the resulting frame
    cv2.imshow('Frame',frame)
    out.write(frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop

print ("in release") 
# When everything done, release the video capture object
cap.release()
print ("done" )
# Closes all the frames
cv2.destroyAllWindows()

