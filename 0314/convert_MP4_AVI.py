import cv2
import numpy as np
import sys

#translate MP4 to avi

cap = cv2.VideoCapture(sys.argv[1])

fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter('0314_4M_EVAN.avi', fourcc, 30, (1920,1080))

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

cap.set(1,60*4) #skip number of frame

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

